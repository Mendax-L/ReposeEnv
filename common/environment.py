"""Shared simulator contract; no policy or training dependencies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import importlib.util
import sys
import numpy as np
from .schema import CAMERA_KEY, get_environment_schema
OBS_DIM = 20
ACTION_DIM = 2
WORKSPACE_SIZE = 512.0


def prepare_native_runtime():
    """Load MuJoCo before Pygame's DLLs when switching tasks on Windows."""
    if sys.platform == "win32" and "mujoco" not in sys.modules and importlib.util.find_spec("mujoco") is not None:
        importlib.import_module("mujoco")


@dataclass(frozen=True)
class StepResult:
    obs: Any
    reward: float
    done: bool
    info: dict[str, Any]
    terminated: bool = False
    truncated: bool = False


class PushEnvironment:
    """One external action/observation contract over a native push environment."""

    name: str
    obs_dim = OBS_DIM
    action_dim = ACTION_DIM
    action_low = (0.0, 0.0)
    action_high = (WORKSPACE_SIZE, WORKSPACE_SIZE)
    observation_layout = "object9_xy_agent_xy/v1"
    action_semantics = "absolute_agent_xy"

    def __init__(self, environment: Any) -> None:
        self.environment = environment
        self._latest_observation: np.ndarray | None = None

    def reset(self, seed: int) -> np.ndarray:
        if hasattr(self.environment, "seed"):
            self.environment.seed(int(seed))
            result = self.environment.reset()
        else:
            result = self.environment.reset(seed=int(seed))
        native_obs = result[0] if isinstance(result, tuple) else result
        observation = self._model_observation(native_obs)
        self._latest_observation = observation.copy()
        return observation

    def step(self, action: np.ndarray | list[float]) -> StepResult:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (self.action_dim,) or not np.isfinite(action_array).all():
            raise ValueError(f"action must be finite with shape ({self.action_dim},), got {action_array}.")
        result = self.environment.step(self._native_action(np.clip(action_array, self.action_low, self.action_high)))
        if len(result) == 5:
            native_obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            native_obs, reward, done, info = result
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)
        observation = self._model_observation(native_obs)
        self._latest_observation = observation.copy()
        return StepResult(
            obs=observation,
            reward=float(reward),
            done=bool(done),
            info=dict(info),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def render(self) -> np.ndarray:
        return np.asarray(self.environment.render(mode="rgb_array"))

    def close(self) -> None:
        self.environment.close()

    @property
    def fps(self) -> int:
        """Control/sampling frequency, independent of physics substeps."""
        frequency = (1.0 / self.environment.dt if self.name == "pushbox3d"
                     else self.environment.control_hz)
        if not np.isclose(frequency, round(frequency)) or frequency <= 0:
            raise ValueError(f"LeRobot requires a positive integer sampling frequency, got {frequency}.")
        return int(round(frequency))

    def lerobot_schema(self) -> dict[str, Any]:
        """Dataset features and units, while contract() retains the IPC schema."""
        shape = (self.environment.render_size, self.environment.render_size, 3)
        schema = get_environment_schema(self.name, image_shape=shape)
        schema["fps"] = self.fps
        return schema

    def lerobot_observation(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        """Read the current state/image without stepping or resetting physics.

        Call after reset() and before step(action) to pair observation_t with
        action_t in a dataset row. After step(), this returns observation_t+1.
        """
        if self._latest_observation is None:
            raise RuntimeError("Call reset(seed) before reading a LeRobot observation.")
        observation = {"observation.state": self._latest_observation.copy()}
        if include_images:
            image = self.render()
            expected_shape = tuple(self.lerobot_schema()["features"][CAMERA_KEY]["shape"])
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(f"Expected RGB uint8 image {expected_shape}, got {image.shape} {image.dtype}.")
            observation[CAMERA_KEY] = image.copy()
        return observation

    def contract(self) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_low": list(self.action_low),
            "action_high": list(self.action_high),
            "observation_layout": self.observation_layout,
            "action_semantics": self.action_semantics,
        }

    def recording_schema(self) -> dict[str, Any]:
        """All available recording modalities, independent of policy inputs."""
        schema = self.lerobot_schema()
        if self.name == "pushbox3d":
            return schema
        agents = ["agent"] if self.name == "pusht" else ["external_agent", "internal_agent"]
        agent_names = [f"{agent}.{axis}" for agent in agents for axis in "xy"]
        keypoint_names = schema["features"]["observation.state"]["names"][:-len(agent_names)]
        fields = {
            "observation.keypoints": (keypoint_names, ["workspace_pixel"] * len(keypoint_names)),
            "observation.agent_pos": (agent_names, ["workspace_pixel"] * len(agent_names)),
            "observation.environment_state": (agent_names + ["block.x", "block.y", "block.angle"],
                                              ["workspace_pixel"] * (len(agent_names) + 2) + ["rad"]),
            "observation.velocity": ([f"{agent}.velocity.{axis}" for agent in agents for axis in "xy"]
                                     + ["block.velocity.x", "block.velocity.y", "block.angular_velocity"],
                                     ["workspace_pixel/s"] * (len(agent_names) + 2) + ["rad/s"]),
            "observation.goal_pose": (["goal.x", "goal.y", "goal.angle"], ["workspace_pixel", "workspace_pixel", "rad"]),
        }
        for key, (names, units) in fields.items():
            schema["features"][key] = {"dtype": "float32", "shape": [len(names)], "names": names}
            schema["feature_semantics"][key] = {"units": units, "description": "Native pre-action simulator values; no policy observation selection is applied."}
        schema["features"]["observation.n_contacts"] = {"dtype": "int64", "shape": [1], "names": ["count"]}
        schema["feature_semantics"]["observation.n_contacts"] = {
            "description": "Native contact-point count per physics substep, rounded up, over the preceding control step; zero after reset."
        }
        return schema

    def recording_observation(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        """Snapshot RGB, keypoints, pose, velocities and contacts together."""
        observation = self.lerobot_observation(include_images=include_images)
        if self.name == "pushbox3d":
            return observation
        native = self.environment
        agents = [native.agent] if self.name == "pusht" else [native.agent, native.internal_agent]
        positions = np.concatenate([np.asarray(agent.position) for agent in agents])
        observation.update({
            "observation.keypoints": self._latest_observation[:-len(positions)].copy(),
            "observation.agent_pos": positions.astype(np.float32),
            "observation.environment_state": np.asarray([*positions, *native.block.position, native.block.angle % (2 * np.pi)], dtype=np.float32),
            "observation.velocity": np.asarray([*(value for agent in agents for value in agent.velocity),
                                                *native.block.velocity, native.block.angular_velocity], dtype=np.float32),
            "observation.goal_pose": np.asarray(native.goal_pose, dtype=np.float32).copy(),
            "observation.n_contacts": np.asarray([native._get_info()["n_contacts"]], dtype=np.int64),
        })
        return observation

    def observe(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        """Read every native modality at the current time without stepping."""
        return self.recording_observation(include_images=include_images)

    def get_keypoints(self) -> np.ndarray:
        """Object keypoint coordinates; implemented by each task."""
        raise NotImplementedError

    def get_agent_positions(self) -> np.ndarray:
        if self._latest_observation is None:
            raise RuntimeError("Call reset(seed) before reading agent positions.")
        if self.name == "pushbox3d":
            return self._latest_observation[37:43].copy()
        position_dim = 4 if self.name == "pushbox2d" else 2
        return self._latest_observation[-position_dim:].copy()

    def teleop_agent(self):
        """Return a controller with act(obs, mouse=..., direction=...)."""
        raise NotImplementedError

    def _native_action(self, action: np.ndarray) -> Any:
        raise NotImplementedError

    def _model_observation(self, native_obs: Any) -> np.ndarray:
        raise NotImplementedError

    def _validate_observation(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.obs_dim,) or not np.isfinite(value).all():
            raise ValueError(f"model observation must be finite with shape ({self.obs_dim},), got {value.shape}.")
        return value


