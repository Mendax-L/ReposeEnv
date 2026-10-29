"""alignbox3d public environment; same API as the other push tasks."""
from __future__ import annotations
from typing import Any
import numpy as np
from common.environment import PushEnvironment
from common.schema import ALIGNBOX3D_OBSERVATION_FIELDS, CAMERA2_KEY

class AlignBox3DEnv(PushEnvironment):
    name = "alignbox3d"
    obs_dim = 66
    action_dim = 14
    action_low = ((-1.0,) * 6 + (0.0,)) * 2
    action_high = (1.0,) * 14
    observation_layout = "stacked_box_state_yaw_cw_grippers_tool_quaternions/v4"
    action_semantics = "dual_ee_world_xyz_rotvec_delta_1mm_0.5deg_gripper_abs_0_1/v4"
    observation_fields = ALIGNBOX3D_OBSERVATION_FIELDS

    def __init__(self, render_size: int = 512) -> None:
        from .native.env import StackedBoxesEnv

        super().__init__(StackedBoxesEnv(render_size=render_size))

    def _native_action(self, action: np.ndarray) -> np.ndarray:
        return action

    def get_end_effector_poses(self) -> np.ndarray:
        """Measured inner/outer XYZ + WXYZ in the shared world frame, shape (2,7)."""
        return self.environment.tip_poses.astype(np.float32)

    def hold_action(self) -> np.ndarray:
        """Return zero tool increments and retain both gripper opening targets."""
        return self.environment.hold_action()

    def reset(self, seed: int, *, options: dict[str, Any] | None = None) -> np.ndarray:
        """Randomize XY/yaw and initial roll/pitch with optional explicit poses."""
        native_obs, _ = self.environment.reset(seed=int(seed), options=options)
        observation = self._model_observation(native_obs)
        self._latest_observation = observation.copy()
        return observation

    def request_finish(self, reason="manual") -> None:
        """End on the next step without moving the robot."""
        self.environment.request_finish(reason)

    def lerobot_schema(self) -> dict[str, Any]:
        from .native.model import ROBOT_MODEL, GRIP_POINTS, GRIP_APPROACH_TILT
        schema = super().lerobot_schema()
        schema["robot_model"] = ROBOT_MODEL
        schema["end_effector_frame"] = {"name": "world", "origin": "MuJoCo scene world origin; lower-box centre XY is (0,0), table surface Z is 0.080 m", "axes": "right-handed shared XYZ for both arms", "pose_order": "XYZ, quaternion WXYZ", "quaternion_maps": "tool frame to world frame", "arm_order": ["inner", "outer"], "state_position_slices": [[37,40],[40,43]], "state_quaternion_slices": [[58,62],[62,66]]}
        schema["grasp"] = {
            "initially_attached": False,
            "automatic_grasp": False,
            "initial_finger_joint_opening_metres": self.environment.gripper_opening,
            "finger_joint_target_limits_metres": self.environment.gripper_limits.tolist(),
            "opening_action": "absolute normalized target; 0 closed, 1 fully open",
            "initial_tool_positions_metres": [list(point) for point in self.environment.initial_tip_positions],
            "initial_tool_target_rotations_world": self.environment.initial_tip_rotations.tolist(),
            "minimum_initial_finger_clearance_metres": self.environment.initial_finger_clearance_m,
            "initial_box_top_height_metres": self.environment.initial_box_top_height,
            "initial_finger_lowest_heights_metres": self.environment.initial_finger_lowest_heights.tolist(),
            "height_check_geometry": "both visible finger meshes and finger collision boxes",
            "location": "side-wall reference sites only; no active grasp constraints",
            "pad_center_offsets_metres": [list(point) for point in GRIP_POINTS],
            "frame": "upper_box local frame; origin at bottom centre",
            "approach_tilt_from_vertical_radians": GRIP_APPROACH_TILT,
        }
        native = self.environment
        schema["initialization"] = {
            "distribution": "seeded box XY/yaw proposals, conditioned on minimum radial/angular offsets; tool parked poses are independent of the box",
            "low": list(native.initial_offset_low),
            "high": list(native.initial_offset_high),
            "min_distance": native.minimum_initial_distance,
            "units": "metres",
            "minimum_distance_applies_to": "randomized resets only",
            "yaw_distribution": "uniform proposals about box centre, rejecting small absolute yaw",
            "yaw_low_radians": native.initial_yaw_low,
            "yaw_high_radians": native.initial_yaw_high,
            "minimum_absolute_yaw_radians": native.minimum_initial_yaw,
            "yaw_positive_direction": "clockwise viewed from above",
            "workspace_rejection": "tool parked poses are checked against Cartesian workspace bounds",
            "yaw_limit_radians": native.yaw_limit,
            "current_initial_yaw_radians": native.initial_yaw,
            "current_yaw_randomized": bool(native.initial_yaw_randomized),
            "box_goal_z_metres": native.control_height,
            "initial_roll_pitch_radians": native.initial_roll_pitch.tolist(),
            "random_roll_pitch_range_radians": [-native.initial_tilt_range, native.initial_tilt_range],
            "tilt_height_rule": "lowest rotated bottom corner starts at the 0.200 m rim plane; gravity may settle the tilt",
            "explicit_offset_priority": True,
            "fixed_offset_metres": native.fixed_offset.tolist(),
            "fixed_yaw_radians": native.fixed_yaw,
            "current_initial_offset_metres": native.initial_offset.tolist(),
            "current_initial_randomized": bool(native.initial_randomized),
            "current_offset_scope": "Snapshot when this schema is read; each episode's actual initial offset is its first observation.state XY and native info.initial_offset.",
            "seed_rule": "reset(seed) reproduces the same sample; new saved episodes use new seeds and discarded attempts reuse their seed",
        }
        schema["episode_lifecycle"] = {
            "control_height_metres": native.control_height, "xy_success_metres": native.xy_tolerance,
            "yaw_success_radians": native.yaw_tolerance,
            "yaw_success_degrees": float(np.degrees(native.yaw_tolerance)), "goal_yaw_radians": 0.0,
            "yaw_success_metric": "absolute yaw error relative to goal",
            "yaw_success_comparison": "<=",
            "xy_success_metric": "radial_xy_distance = sqrt((upper.x - lower.x)^2 + (upper.y - lower.y)^2)",
            "xy_success_comparison": "<=",
            "height_tolerance_metres": native.height_tolerance, "success_hold_seconds": 0.5,
            "height_semantics": {
                "lifted_height": "Actual box Z minus its original resting Z (0.200 m).",
                "height_error": "Actual box Z minus the fixed supported goal height (0.200 m).",
            },
            "automatic_vertical_motion": False,
            "automatic_alignment_completion": native.terminate_on_alignment,
            "initially_attached": False,
            "phases": ["control", "complete", "failed"],
        }
        if not native.terminate_on_alignment:
            schema["termination"] = "Automatic box-alignment completion is disabled for this reset. The caller manages task completion and may enable alignment verification after releasing the object; manual/time-limit completion without success is truncated. No automatic grasp or vertical motion."
        return schema

    def _model_observation(self, native_obs: Any) -> np.ndarray:
        flattened = np.concatenate([np.asarray(native_obs[key]).reshape(-1) for key in self.observation_fields])
        return self._validate_observation(flattened)

    def render(self) -> np.ndarray:
        return np.asarray(self.environment.render())

    def render_preview(self) -> dict[str, np.ndarray]:
        """Native-resolution display streams, independent of dataset settings."""
        return {"Overview": self.environment.render(camera="overview", render_size=512),
                "Closeup": self.environment.render(camera="closeup", render_size=512)}

    def lerobot_observation(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        observation = super().lerobot_observation(include_images=include_images)
        if include_images:
            image = np.asarray(self.environment.render(camera="closeup"))
            expected_shape = tuple(self.lerobot_schema()["features"][CAMERA2_KEY]["shape"])
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(f"Expected RGB uint8 camera2 image {expected_shape}, got {image.shape} {image.dtype}.")
            observation[CAMERA2_KEY] = image.copy()
        return observation

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
        schema = self.lerobot_schema()
        state_names = schema["features"]["observation.state"]["names"]
        fields = {
            "observation.keypoints": ([f"object.corner_{i}.{a}" for i in range(8) for a in "xyz"], ["m"] * 24),
            "observation.agent_pos": ([f"{arm}.tip.{a}" for arm in ("inner", "outer") for a in "xyz"], ["m"] * 6),
            "observation.environment_state": (state_names, schema["feature_semantics"]["observation.state"]["units"]),
            "observation.velocity": (state_names[7:13], ["m/s"] * 3 + ["rad/s"] * 3),
            "observation.goal_pose": (["goal.x", "goal.y", "goal.z", "goal.qw", "goal.qx", "goal.qy", "goal.qz"], ["m"] * 3 + ["unitless"] * 4),
            "observation.joint_positions": (state_names[13:25], ["rad"] * 12),
            "observation.joint_velocities": (state_names[25:37], ["rad/s"] * 12),
            "observation.tip_targets": (state_names[43:49], ["m"] * 6),
            "observation.contact_forces": (state_names[52:54], ["N"] * 2),
            "observation.gripper_positions": (state_names[54:56], ["m"] * 2),
            "observation.gripper_targets": (state_names[56:58], ["m"] * 2),
            "observation.end_effector_poses": ([f"{arm}.pose.{axis}" for arm in ("inner", "outer") for axis in ("x","y","z","qw","qx","qy","qz")], (["m"]*3 + ["unitless"]*4)*2),
            "observation.yaw": (["box.yaw"], ["rad"]),
            "observation.tip_target_rotations": ([f"{arm}.tip_target_rotation.{row}{column}" for arm in ("inner", "outer") for row in range(3) for column in range(3)], ["unitless"] * 18),
        }
        for key, (names, units) in fields.items():
            schema["features"][key] = {"dtype": "float32", "shape": [len(names)], "names": names}
            schema["feature_semantics"][key] = {"units": units, "description": "Native pre-action world/robot state; all modalities are recorded."}
        schema["feature_semantics"]["observation.yaw"]["description"] = "Box-centred yaw in radians, clockwise positive and counterclockwise negative when viewed from above. Equal to observation.state[51]; native world quaternions remain right-handed."
        schema["feature_semantics"]["observation.end_effector_poses"]["description"] = "Measured absolute inner then outer tool XYZ + quaternion WXYZ, both in the SAME shared world frame. Not target poses or separate arm-local poses."
        for field in ("observation.gripper_positions", "observation.gripper_targets"):
            schema["feature_semantics"][field]["description"] = "Inner then outer driven-finger slide position/target in metres; 0..0.044 m nominal range. Not the physical gap between finger pads."
        schema["features"]["observation.n_contacts"] = {"dtype": "int64", "shape": [1], "names": ["count"]}
        schema["feature_semantics"]["observation.n_contacts"] = {"description": "MuJoCo contact count at this observation time; not contact-force magnitude."}
        schema["goal_pose"] = [0, 0, self.environment.control_height, 1, 0, 0, 0]
        schema["features"]["observation.phase"] = {"dtype": "int64", "shape": [1], "names": ["phase"]}
        schema["feature_semantics"]["observation.phase"] = {
            "description": "Episode stage before action: 0 control, 1 complete, 2 failed."
        }
        return schema

    def recording_observation(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        observation = self.lerobot_observation(include_images=include_images)
        state = self._latest_observation
        observation.update({
            "observation.keypoints": self.get_keypoints().reshape(-1),
            "observation.agent_pos": state[37:43].copy(),
            "observation.environment_state": state.copy(),
            "observation.velocity": state[7:13].copy(),
            "observation.goal_pose": np.asarray([0, 0, self.environment.control_height, 1, 0, 0, 0], dtype=np.float32),
            "observation.joint_positions": state[13:25].copy(),
            "observation.joint_velocities": state[25:37].copy(),
            "observation.tip_targets": state[43:49].copy(),
            "observation.contact_forces": state[52:54].copy(),
            "observation.gripper_positions": state[54:56].copy(),
            "observation.gripper_targets": state[56:58].copy(),
            "observation.end_effector_poses": self.get_end_effector_poses().reshape(-1),
            "observation.yaw": state[51:52].copy(),
            "observation.tip_target_rotations": self.environment.tip_target_rotations.reshape(-1).astype(np.float32),
            "observation.n_contacts": np.asarray([self.environment.data.ncon], dtype=np.int64),
            "observation.phase": np.asarray([
                ["control", "complete", "failed"].index(self.environment.phase)
            ], dtype=np.int64),
        })
        return observation

