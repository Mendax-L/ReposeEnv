"""Record supplied actions in a pushenv environment as a LeRobot dataset.

Example (the actions file must contain real user/controller commands)::

    python record.py --env pusht --actions commands.npy \
        --output training_data/pusht_recorded --task "Push the T into its goal"

``record_episode`` expects an environment that has already been reset. Each row
contains observation_t, the clipped action_t, and the resulting next.* fields.
This module records commands; it does not label them as expert demonstrations.
"""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np



TRANSITION_FEATURES = {
    "next.reward": {"dtype": "float32", "shape": [1], "names": ["reward"]},
    "next.done": {"dtype": "bool", "shape": [1], "names": ["done"]},
    "next.terminated": {"dtype": "bool", "shape": [1], "names": ["terminated"]},
    "next.truncated": {"dtype": "bool", "shape": [1], "names": ["truncated"]},
    "next.success": {"dtype": "bool", "shape": [1], "names": ["success"]},
    "next.success_known": {"dtype": "bool", "shape": [1], "names": ["success_known"]},
}


def recording_schema(environment: Any) -> dict[str, Any]:
    """Declare transition fields before constructing a LeRobotWriter."""
    schema_factory = getattr(environment, "recording_schema", environment.lerobot_schema)
    schema = copy.deepcopy(schema_factory())
    schema["features"].update(copy.deepcopy(TRANSITION_FEATURES))
    semantics = schema.setdefault("feature_semantics", {})
    descriptions = {
        "next.reward": "Reward returned after applying this row's action.",
        "next.done": "Episode boundary: native termination/truncation, manual save/exit, action-source exhaustion, or collector max_steps.",
        "next.terminated": "Native environment termination; collection limits never set this flag.",
        "next.truncated": "Native time limit or collector boundary (manual save/exit, max_steps or action-source exhaustion) without native termination.",
        "next.success": "Native info.success when available; otherwise 2-D info.coverage > native success_threshold (default 0.95). False if unknown; consult next.success_known.",
        "next.success_known": "Whether success is explicitly available from native info.success or the 2-D coverage criterion.",
    }
    for name, description in descriptions.items():
        semantics[name] = {"description": description}
    schema["recording_alignment"] = (
        "State and RGB are captured immediately before action_t; next.* describe "
        "the resulting transition. Commands are clipped to the adapter action "
        "bounds before both execution and storage. The final row is truncated "
        "when manual save/exit, the action source or max_steps ends an otherwise active episode."
    )
    return schema


def _success(environment: Any, info: dict[str, Any]) -> tuple[bool, bool]:
    if "success" in info:
        return bool(info["success"]), True
    if environment.name in ("pusht", "pushbox2d") and "coverage" in info:
        coverage = float(info["coverage"])
        if not np.isfinite(coverage):
            raise ValueError("Native coverage must be finite when provided.")
        native = getattr(environment, "environment", None)
        threshold = float(getattr(native, "success_threshold", 0.95))
        return coverage > threshold, True
    return False, False


class EpisodeRecorder:
    """Incremental recording shared by live teleoperation and action replay."""

    def __init__(self, environment: Any, *, buffer_dir=None):
        self.environment = environment
        self.features = recording_schema(environment)["features"]
        self.rows = {name: [] for name in self.features}
        self.last_transition = None
        self._scratch = TemporaryDirectory(prefix=".recording-", dir=buffer_dir) if buffer_dir is not None else None
        self._video_files = {}
        self._video_paths = {}

    @property
    def frames(self) -> int:
        return len(self.rows["action"])

    def step(self, supplied_action):
        environment = self.environment
        if self.last_transition is not None and (self.last_transition.done or self.last_transition.terminated or self.last_transition.truncated):
            raise RuntimeError("Cannot step a finished episode; save or reset it first.")
        action = np.asarray(supplied_action, dtype=np.float32)
        if action.shape != (environment.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f"Action {self.frames} must be finite with shape ({environment.action_dim},), got {action.shape}.")
        action = np.clip(action, environment.action_low, environment.action_high).astype(np.float32)
        observe = getattr(environment, "recording_observation", environment.lerobot_observation)
        snapshot = {name: np.asarray(value).copy() for name, value in observe().items()}
        transition = environment.step(action)
        if not np.isfinite(transition.reward):
            raise ValueError(f"Non-finite reward at action {self.frames}.")
        success, success_known = _success(environment, transition.info)
        snapshot["action"] = action.copy()
        snapshot.update({
            "next.reward": np.asarray([transition.reward], dtype=np.float32),
            "next.done": np.asarray([transition.done or transition.terminated or transition.truncated], dtype=bool),
            "next.terminated": np.asarray([transition.terminated], dtype=bool),
            "next.truncated": np.asarray([transition.truncated], dtype=bool),
            "next.success": np.asarray([success], dtype=bool),
            "next.success_known": np.asarray([success_known], dtype=bool),
        })
        if set(snapshot) != set(self.rows):
            raise ValueError(f"Observation/transition features differ from recording schema: {set(snapshot) ^ set(self.rows)}.")
        for name in self.rows:
            value = snapshot[name]
            if self._scratch is not None and self.features[name]["dtype"] == "video":
                if value.dtype != np.uint8 or list(value.shape) != self.features[name]["shape"]:
                    raise ValueError(f"{name} must match the RGB recording schema.")
                if name not in self._video_files:
                    path = Path(self._scratch.name) / f"camera-{len(self._video_files)}.rgb"
                    self._video_paths[name] = path
                    self._video_files[name] = path.open("wb")
                self._video_files[name].write(value.tobytes())
            else:
                self.rows[name].append(value)
        self.last_transition = transition
        return transition

    def save(self, writer: Any, stop_reason: str) -> dict[str, Any]:
        if not self.frames:
            raise ValueError("Cannot record an empty action source.")
        mapped = []
        batch = {}
        try:
            for name, values in self.rows.items():
                if name in self._video_files:
                    self._video_files[name].flush()
                    value = np.memmap(self._video_paths[name], dtype=np.uint8, mode="r",
                                      shape=(self.frames, *self.features[name]["shape"]))
                    mapped.append(value)
                    batch[name] = value
                else:
                    batch[name] = np.stack(values, axis=0)
            if stop_reason != "environment":
                batch["next.truncated"][-1, 0] = True
                batch["next.done"][-1, 0] = True
            result = {
                "frames": self.frames, "stop_reason": stop_reason,
                **{name: bool(batch[f"next.{name}"][-1, 0])
                   for name in ("terminated", "truncated", "success", "success_known")},
            }
            writer.write_episode(batch)
            # Flush before releasing memory maps; the writer may otherwise keep
            # pending references until a later episode or close().
            if self._scratch is not None:
                writer.checkpoint()
            return result
        except BaseException:
            # A failed encoder may still hold views into our temporary files.
            # Release the writer's references before closing their memory maps.
            abort = getattr(writer, "abort", None)
            if abort is not None:
                abort()
            raise
        finally:
            for value in mapped:
                value._mmap.close()
            if self._scratch is not None:
                self.close()

    def close(self):
        for handle in self._video_files.values():
            handle.close()
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None


def record_episode(
    environment: Any,
    actions: Iterable[np.ndarray | list[float]],
    writer: Any,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Execute one episode and write aligned feature batches to ``writer``.

    Construct the writer with ``recording_schema(environment)`` first and reset
    the environment before calling. The caller owns closing the writer and the
    environment, allowing several separately reset episodes in one dataset.
    """
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None.")
    recorder = EpisodeRecorder(environment, buffer_dir=getattr(writer, "root", None))
    iterator = iter(actions)
    if environment.name == "pushbox3d" and hasattr(environment, "request_finish"):
        finishing = False
        stop_reason = "environment"
        try:
            while True:
                if max_steps is not None and recorder.frames >= max_steps and not finishing:
                    environment.request_finish("max_steps")
                    finishing = True
                if environment.environment.phase == "control" and not finishing:
                    try:
                        supplied_action = next(iterator)
                    except StopIteration:
                        environment.request_finish("actions_exhausted")
                        finishing = True
                        supplied_action = np.zeros(environment.action_dim, dtype=np.float32)
                else:
                    supplied_action = np.zeros(environment.action_dim, dtype=np.float32)
                transition = recorder.step(supplied_action)
                if transition.done:
                    break
            return recorder.save(writer, stop_reason)
        finally:
            recorder.close()
    step_count = 0
    stop_reason = "actions_exhausted"

    try:
        while max_steps is None or step_count < max_steps:
            try:
                supplied_action = next(iterator)
            except StopIteration:
                break
            transition = recorder.step(supplied_action)
            step_count += 1
            if transition.done or transition.terminated or transition.truncated:
                stop_reason = "environment"
                break
        else:
            stop_reason = "max_steps"
        return recorder.save(writer, stop_reason)
    finally:
        recorder.close()


def record_straight_episode(
    environment: Any,
    writer: Any,
    max_steps: int | None = None,
    speed_m_s: float = 0.02,
    yaw_speed_rad_s: float = float(np.deg2rad(10.)),
) -> dict[str, Any]:
    """Record a physical PushBox3D cycle using one fixed XY command direction.

    The centre follows a fixed XY line while yaw turns towards zero. Opening
    lift, landing and settling remain native physics stages. Actual box motion
    is measured, never overwritten.
    """
    if environment.name != "pushbox3d" or environment.action_dim != 3:
        raise ValueError("Straight-line collection requires PushBox3D with centre XY/yaw actions.")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None.")
    if not np.isfinite(speed_m_s) or not 0 < speed_m_s <= 0.001 * environment.fps:
        raise ValueError("speed_m_s must be positive and at most 0.001 * fps metres per second.")
    if getattr(environment, "_latest_observation", None) is None:
        raise RuntimeError("Reset the environment before straight-line collection.")

    from pushBox3D.native.model import GRIP_POINTS

    native = environment.environment
    if not np.isfinite(yaw_speed_rad_s) or not 0 < yaw_speed_rad_s <= native.yaw_action_scale * environment.fps:
        raise ValueError("yaw_speed_rad_s must be positive and no larger than the yaw action speed limit.")
    if native.phase in ("complete", "failed"):
        raise RuntimeError("Reset the finished environment before collecting another episode.")
    goal_grips = np.asarray(GRIP_POINTS, dtype=np.float64)[:, :2]
    initial_xy = native.pose[0][:2].copy()
    initial_yaw = float(native.pose[1])
    goal_xy = np.zeros(2, dtype=np.float64)
    box_line = goal_xy - initial_xy
    box_line_squared = float(box_line @ box_line)
    control_start = direction = None
    command_distance = speed_m_s / environment.fps
    limit = max_steps if max_steps is not None else native.max_steps
    tail_limit = int(np.ceil(3 * native.stage_timeout_seconds * environment.fps)) + 3
    recorder = EpisodeRecorder(environment, buffer_dir=getattr(writer, "root", None))
    previous_xy = initial_xy.copy()
    max_cross_track = actual_path_length = 0.0
    phase_counts: dict[str, int] = {}
    control_steps = 0
    finish_at = None
    stop_reason = "environment"
    bounded_tail_exhausted = False
    try:
        while True:
            if recorder.frames >= limit and finish_at is None:
                environment.request_finish("max_steps" if max_steps is not None else "time_limit")
                finish_at = recorder.frames
            if native.finish_requested and finish_at is None:
                finish_at = recorder.frames
            if finish_at is not None and recorder.frames - finish_at >= tail_limit:
                stop_reason = "tail_timeout"
                bounded_tail_exhausted = True
                break

            phase = native.phase
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            action = np.zeros(environment.action_dim, dtype=np.float32)
            if phase == "control" and finish_at is None:
                center = native.box_target_center[:2]
                if control_start is None:
                    control_start = center.copy()
                    distance = float(np.linalg.norm(control_start))
                    direction = -control_start / distance if distance > 1e-12 else np.zeros(2)
                # Only the remaining scalar distance changes. The direction
                # and goal stay fixed even if the free box lags its targets.
                remaining = max(0.0, float((-center) @ direction))
                action[:2] = direction * min(command_distance, remaining) / 0.001
                yaw_step = min(yaw_speed_rad_s / environment.fps, native.yaw_action_scale)
                action[2] = np.clip(-native.box_target_yaw, -yaw_step, yaw_step) / native.yaw_action_scale
                control_steps += 1

            transition = recorder.step(action)
            actual_xy = native.pose[0][:2]
            actual_path_length += float(np.linalg.norm(actual_xy - previous_xy))
            previous_xy = actual_xy.copy()
            projection = (np.clip(float((actual_xy - initial_xy) @ box_line) / box_line_squared, 0, 1)
                          if box_line_squared > 1e-24 else 0.0)
            closest = initial_xy + projection * box_line
            max_cross_track = max(max_cross_track, float(np.linalg.norm(actual_xy - closest)))
            if transition.done or transition.terminated or transition.truncated:
                break

        info = recorder.last_transition.info
        result = recorder.save(writer, stop_reason)
        final_center = native.box_target_center[:2]
        result.update({
            "initial_offset_m": initial_xy.tolist(),
            "initial_yaw_rad": initial_yaw,
            "initial_yaw_deg": float(np.degrees(initial_yaw)),
            "yaw_convention": "clockwise_positive_viewed_from_above",
            "final_xy_error_m": float(info["xy_error"]),
            "final_yaw_error_rad": float(info["yaw_error"]),
            "final_yaw_error_deg": float(np.degrees(info["yaw_error"])),
            "final_height_m": float(info["height"]),
            "landing_height_error_m": float(info["landing_height_error"]),
            "max_cross_track_error_m": max_cross_track,
            "actual_xy_path_length_m": actual_path_length,
            "final_phase": str(info["phase"]),
            "finish_reason": info.get("finish_reason"),
            "failure": bool(info["failure"]),
            "failure_reason": info.get("failure_reason"),
            "bounded_tail_exhausted": bounded_tail_exhausted,
            "phase_frames": phase_counts,
            "controller": {
                "name": "fixed_direction_straight_xy_with_center_yaw",
                "speed_m_s": float(speed_m_s),
                "yaw_speed_rad_s": float(yaw_speed_rad_s),
                "yaw_action_scale_rad": native.yaw_action_scale,
                "goal_yaw_rad": 0.,
                "final_target_yaw_rad": native.box_target_yaw,
                "fps": environment.fps,
                "step_distance_m": command_distance,
                "action_scale_m": 0.001,
                "control_start_xy_m": None if control_start is None else control_start.tolist(),
                "goal_xy_m": goal_xy.tolist(),
                "direction_xy": None if direction is None else direction.tolist(),
                "goal_tip_targets_xy_m": goal_grips.tolist(),
                "final_target_offset_xy_m": final_center.tolist(),
                "endpoint_reached": bool(np.linalg.norm(final_center) <= 1e-8 and abs(native.box_target_yaw) <= 1e-8),
                "control_steps": control_steps,
                "max_steps": max_steps,
                "tail_limit_steps": tail_limit,
            },
        })
        return result
    finally:
        recorder.close()


def load_actions(path: Path) -> np.ndarray:
    """Read an N-by-action_dim .npy or the 'action' array from a .npz file."""
    if path.suffix.lower() not in (".npy", ".npz"):
        raise ValueError("Actions must be stored in a .npy or .npz file.")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            if "action" not in loaded.files:
                raise ValueError("The .npz action source must contain an 'action' array.")
            actions = np.asarray(loaded["action"], dtype=np.float32)
    else:
        actions = np.asarray(loaded, dtype=np.float32)
    if actions.ndim != 2 or len(actions) == 0 or not np.isfinite(actions).all():
        raise ValueError("The action source must be a nonempty finite 2-D [frames, action_dim] array.")
    return actions

