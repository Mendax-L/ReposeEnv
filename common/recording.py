"""Record supplied actions in a ReposeEnv environment as a LeRobot dataset.

Example (the actions file must contain real user/controller commands)::

    python -m common.recording --env pusht --actions commands.npy \
        --output training_data/pusht_recorded --task "Push the T into its goal"

``record_episode`` expects an environment that has already been reset. Each row
contains observation_t, the clipped action_t, and the resulting next.* fields.
This module records commands; it does not label them as expert demonstrations.
"""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

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
    if environment.name == "pusht" and "coverage" in info:
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

    def save(self, writer: Any, stop_reason: str, *, task: str | Sequence[str] | None = None) -> dict[str, Any]:
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
            writer.write_episode(batch, task=task)
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
    if environment.name == "alignbox3d" and hasattr(environment, "request_finish"):
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
                        supplied_action = environment.hold_action()
                else:
                    supplied_action = environment.hold_action()
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


TASKS = {
    "pusht": "Push the T into its goal",
    "alignbox3d": "Control both ARX L5 end effectors to align the upper box through physical contact",
}


def main(default_env="pusht", argv=None):
    """Collect, replay or validate data with ``python -m common.recording``."""
    import argparse
    from datetime import datetime, timedelta, timezone
    import hashlib
    import json

    from common.teleoperation import DemonstrationSession, run_window
    from environment import make_environment, SUPPORTED_ENVIRONMENTS

    parser = argparse.ArgumentParser(description="Human collection, action replay and LeRobot dataset validation.")
    parser.add_argument("--env", choices=SUPPORTED_ENVIRONMENTS, default=default_env)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--episodes", type=int, help="Human: 0/unset means unlimited.")
    parser.add_argument("--render-size", type=int, help="Recorded camera width; default 512 for AlignBox3D, 96 for 2-D.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--actions", type=Path, help="Replay one episode of controller .npy/.npz actions instead of human control.")
    source.add_argument("--validate", type=Path, metavar="DATASET", help="Check an existing LeRobot dataset without starting a simulator.")
    args = parser.parse_args(argv)
    if args.validate:
        from common.lerobot import validate_dataset
        print(json.dumps(validate_dataset(args.validate), indent=2))
        return
    args.episodes = 0 if args.episodes is None else args.episodes
    args.render_size = (512 if args.env == "alignbox3d" else 96) if args.render_size is None else args.render_size
    max_steps = args.max_steps if args.max_steps is not None else (6000 if args.env == "alignbox3d" else 3000)
    if max_steps < 1 or args.episodes < 0 or not 0 <= args.seed < 2**63:
        parser.error("max-steps must be positive; episodes nonnegative; seed in [0, 2**63)")
    if args.render_size < 16 or args.render_size % 2:
        parser.error("render-size must be an even integer >= 16")
    if args.actions and args.episodes not in (0, 1):
        parser.error("--actions supplies exactly one episode")
    task = args.task if args.task is not None else TASKS[args.env]
    if not task.strip():
        parser.error("task must contain a task description")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    kind = "replay" if args.actions else "human"
    output = args.output or Path(__file__).resolve().parents[1] / "training_data/lerobot" / args.env / f"{kind}-{stamp}"
    actions = load_actions(args.actions) if args.actions else None
    from common.lerobot import LeRobotWriter, validate_dataset
    environment = make_environment(args.env, render_size=args.render_size)
    if hasattr(environment.environment, "render_action"):
        environment.environment.render_action = False
    session = None
    writer = None
    replay_saved = False
    try:
        if actions is not None and actions.shape[1] != environment.action_dim:
            parser.error(f"{args.env} requires {environment.action_dim} action channels")
        environment.reset(args.seed)
        provenance = {
            "kind": "supplied_controller_actions" if args.actions else "human_teleoperation",
            "human_demonstration": not args.actions,
            "environment": args.env, "initial_seed": args.seed,
            "episode_seed_rule": "initial_seed + saved episode_index; discarded attempts reuse the seed",
            "modalities": "all available native observations and RGB",
            "observation_selection": "deferred to the training algorithm",
            "max_steps": max_steps, "demonstration_quality": "success is recorded, not assumed",
            "rgb_action_markers": False,
            "render_size": args.render_size,
            "environment_contract": environment.contract(),
            "initialization": environment.lerobot_schema().get("initialization"),
        }
        if args.actions:
            provenance.update(action_source=str(args.actions.resolve()),
                              action_source_sha256=hashlib.sha256(args.actions.read_bytes()).hexdigest())
        writer = LeRobotWriter(output, recording_schema(environment), task, provenance=provenance,
                               max_frames_per_file=1)
        print(f"Dataset: {output.resolve()}", flush=True)
        if actions is not None:
            environment.reset(args.seed)
            result = record_episode(environment, actions, writer, max_steps=max_steps)
            replay_saved = True
            print(json.dumps(result), flush=True)
        else:
            session = DemonstrationSession(environment, writer, args.seed, max_steps)
            print(("Space starts/pauses; 1/2/3 select inner/outer/both tools; arrows: XY; PgUp/Dn: Z; Q/E: yaw; I/K: rx; J/L: ry; O/P: open/close grippers; Enter saves; Esc saves and quits."
                   if args.env == "alignbox3d" else
                   "Space starts/pauses; Enter saves; R discards; Esc saves partial and quits."), flush=True)
            try:
                run_window(session, args.episodes)
            except KeyboardInterrupt:
                if writer.aborted:
                    raise
                session.finish_and_save("keyboard_interrupt")
    finally:
        try:
            if session is not None:
                session.close()
            if writer is not None and (replay_saved or (session is not None and session.saved)):
                writer.close()
        finally:
            environment.close()
    if replay_saved or (session is not None and session.saved):
        print(json.dumps(validate_dataset(output), indent=2))
    else:
        print("No demonstrations saved; no dataset was finalized.")


if __name__ == "__main__":
    main()
