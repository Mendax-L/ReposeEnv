"""alignbox3d public environment; same API as the other push tasks."""
from __future__ import annotations
from typing import Any
import numpy as np
from common.environment import PushEnvironment
from common.schema import CAMERA2_KEY
from .geometry import quaternion, rotation_vector

class AlignBox3DEnv(PushEnvironment):
    name = "alignbox3d"
    task_description = "请抓取上箱，抬起并与下箱对齐，然后平稳放下、松开夹爪并撤离。"
    obs_dim = 20
    action_dim = 14
    action_low = ((-1.0,) * 6 + (0.0,)) * 2
    action_high = (1.0,) * 14
    observation_layout = "left_right_end_effector_pos_rot6d_gripper/v1"
    action_semantics = "dual_ee_delta_xyz_local_roll_pitch_yaw_gripper_abs_0_1/v4"

    def __init__(self, render_size: int = 512) -> None:
        from .native.env import StackedBoxesEnv

        super().__init__(StackedBoxesEnv(render_size=render_size))

    def _native_action(self, action: np.ndarray) -> np.ndarray:
        commands = np.asarray([np.concatenate([
            action[self.action_groups[f"action.{side}_end_effector_delta_{suffix}"]]
            for suffix in ("xyz", "rpy")]) for side in ("left", "right")])
        native = self.environment.hold_action().reshape(2, 7)
        native[:, :3] = commands[:, :3]
        for arm, side in enumerate(("left", "right")):
            native[arm, 6] = action[self.action_groups[f"action.{side}_gripper"]][0]
        for arm, values in enumerate(commands[:, 3:6]):
            roll, pitch, yaw = values * self.environment.rotation_action_scale
            cx, sx = np.cos(roll), np.sin(roll)
            cy, sy = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            local_delta = (np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
                           @ np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
                           @ np.array([[1., 0., 0.], [0., cx, -sx], [0., sx, cx]]))
            target = self.environment.tip_target_rotations[arm]
            world_delta = target @ local_delta @ target.T
            native[arm, 3:6] = rotation_vector(world_delta) / self.environment.rotation_action_scale
        return native.ravel()

    def native_action_to_interface(self, action: np.ndarray) -> np.ndarray:
        """Convert world-rotvec expert commands to named XYZ/RPY/gripper actions."""
        commands = np.asarray(action, dtype=float).reshape(2, 7)
        public = np.zeros((2, 6), dtype=np.float32)
        public[:, :3] = commands[:, :3]
        for arm in range(2):
            world_delta = self.environment._rotation_increment(
                commands[arm, 3:6] * self.environment.rotation_action_scale)
            target = self.environment.tip_target_rotations[arm]
            local = target.T @ world_delta @ target
            roll = np.arctan2(local[2, 1], local[2, 2])
            pitch = np.arcsin(np.clip(-local[2, 0], -1., 1.))
            yaw = np.arctan2(local[1, 0], local[0, 0])
            public[arm, 3:6] = np.array([roll, pitch, yaw]) / self.environment.rotation_action_scale
        fields = {f"action.{side}_end_effector_delta_{suffix}": public[arm, offset:offset + 3]
                  for side, arm in (("left", 0), ("right", 1))
                  for suffix, offset in (("xyz", 0), ("rpy", 3))}
        fields.update({f"action.{side}_gripper": commands[arm, 6:7]
                       for arm, side in enumerate(("left", "right"))})
        return np.clip(np.concatenate([fields[key] for key in self._interface_config["action"]]),
                       self.action_low, self.action_high).astype(np.float32)

    def get_end_effector_poses(self) -> np.ndarray:
        """Measured inner/outer XYZ + WXYZ in the shared world frame, shape (2,7)."""
        return self.environment.tip_poses.astype(np.float32)

    def hold_action(self) -> np.ndarray:
        """Keep both target poses and absolute gripper targets unchanged."""
        action = np.zeros(self.action_dim, dtype=np.float32)
        native = self.environment.hold_action().reshape(2, 7)
        for arm, side in enumerate(("left", "right")):
            action[self.action_groups[f"action.{side}_gripper"]] = native[arm, 6]
        return action

    def reset(self, seed: int, *, options: dict[str, Any] | None = None) -> np.ndarray:
        """Randomize XY/yaw and initial roll/pitch with optional explicit poses."""
        native_obs, _ = self.environment.reset(seed=int(seed), options=options)
        self.instruction = self.task_description
        observation = self._model_observation(native_obs)
        self._latest_observation = observation.copy()
        return observation

    def request_finish(self, reason="manual") -> None:
        """End on the next step without moving the robot."""
        self.environment.request_finish(reason)

    def lerobot_schema(self) -> dict[str, Any]:
        schema = super().lerobot_schema()
        schema["interface"] = self._interface_config
        schema["rotation_suffixes"] = {
            "rot6d": "first two rotation-matrix columns",
            "wxyz": "unit quaternion in WXYZ order",
            "rotm": "row-major 3x3 rotation matrix",
            "rpy": "local delta in roll, pitch, yaw order; R_next = R_previous @ Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        }
        return schema

    def _model_observation(self, native_obs: Any) -> np.ndarray:
        metadata = self.interface_observation(include_images=False)
        return self._validate_observation(np.concatenate([
            metadata[key] for key in self._interface_config["observation"]["robot_state"]]))

    def render(self) -> np.ndarray:
        return np.asarray(self.environment.render())

    def render_preview(self) -> dict[str, np.ndarray]:
        """Native-resolution display streams, independent of dataset settings."""
        return {"Overview": self.environment.render(camera="overview", render_size=512),
                "Closeup": self.environment.render(camera="closeup", render_size=512)}

    def lerobot_observation(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        if self._latest_observation is None:
            raise RuntimeError("Call reset(seed) before reading an observation.")
        observation = {"observation.state": self._latest_observation.copy()}
        if include_images and CAMERA2_KEY in self.lerobot_schema()["features"]:
            image = np.asarray(self.environment.render(camera="closeup"))
            expected_shape = tuple(self.lerobot_schema()["features"][CAMERA2_KEY]["shape"])
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(f"Expected RGB uint8 camera2 image {expected_shape}, got {image.shape} {image.dtype}.")
            observation[CAMERA2_KEY] = image.copy()
        return observation

    def interface_observation(self, *, include_images: bool = True) -> dict[str, Any]:
        """Return exactly the observation fields declared by alignbox3d.yml."""
        rotations = self.environment.tip_rotations
        positions = self.environment.tip_positions.astype(np.float32)
        grippers = self.environment.gripper_positions.astype(np.float32)
        result = {}
        for side, arm in (("left", 0), ("right", 1)):
            result[f"robot.{side}_end_effector_pos"] = positions[arm].copy()
            result[f"robot.{side}_end_effector_rot6d"] = rotations[arm, :, :2].T.reshape(-1).astype(np.float32)
            result[f"robot.{side}_end_effector_wxyz"] = quaternion(rotations[arm]).astype(np.float32)
            result[f"robot.{side}_end_effector_rotm"] = rotations[arm].reshape(-1).astype(np.float32)
            result[f"robot.{side}_gripper"] = grippers[arm:arm + 1].copy()
        result["instruction"] = self.instruction
        if include_images:
            result["images.camera2"] = np.asarray(self.environment.render(camera="closeup")).copy()
        declared = [field for fields in self._interface_config["observation"].values() for field in fields]
        return {field: result[field] for field in declared if include_images or not field.startswith("images.")}

    def get_keypoints(self) -> np.ndarray:
        """Eight outer box corners, world XYZ, from the rigid-body transform."""
        from .native.model import BOX_SIZE
        if self._latest_observation is None:
            raise RuntimeError("Call reset(seed) before reading keypoints.")
        native = self.environment
        corners = np.asarray([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], dtype=np.float64)
        corners *= np.asarray(BOX_SIZE) / 2
        # Native body origin is at the outer floor, not the box centre.
        corners[:, 2] += BOX_SIZE[2] / 2
        rotation = native.data.xmat[native.box_id].reshape(3, 3)
        return (corners @ rotation.T + native.data.xpos[native.box_id]).astype(np.float32)

    def teleop_agent(self):
        from .teleop import create_teleop
        return create_teleop(self)

    def recording_schema(self) -> dict[str, Any]:
        return self.lerobot_schema()

    def recording_observation(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        return self.lerobot_observation(include_images=include_images)

