"""pushbox3d public environment; same API as the other push tasks."""
from __future__ import annotations
from typing import Any
import numpy as np
from common.environment import PushEnvironment
from common.schema import BOX3D_OBSERVATION_FIELDS, CAMERA2_KEY

class PushBox3DEnv(PushEnvironment):
    name = "pushbox3d"
    obs_dim = 54
    action_dim = 3
    action_low = (-1.0,) * 3
    action_high = (1.0,) * 3
    observation_layout = "stacked_box_state_yaw_cw/v2"
    action_semantics = "box_center_xy_yaw_cw_delta_1mm_0.5deg_auto_lift_20mm/v2"
    observation_fields = BOX3D_OBSERVATION_FIELDS

    def __init__(self, render_size: int = 512) -> None:
        from .native.env import StackedBoxesEnv

        super().__init__(StackedBoxesEnv(render_size=render_size))

    def _native_action(self, action: np.ndarray) -> np.ndarray:
        return action

    def reset(self, seed: int, *, options: dict[str, Any] | None = None) -> np.ndarray:
        """Randomize XY/yaw reproducibly, with optional explicit/fixed poses."""
        native_obs, _ = self.environment.reset(seed=int(seed), options=options)
        observation = self._model_observation(native_obs)
        self._latest_observation = observation.copy()
        return observation

    def request_finish(self, reason="manual") -> None:
        """Queue physical landing; completion is reported by later step() calls."""
        self.environment.request_finish(reason)

    def lerobot_schema(self) -> dict[str, Any]:
        from .native.model import ROBOT_MODEL, GRIP_POINTS, GRIP_APPROACH_TILT
        schema = super().lerobot_schema()
        schema["robot_model"] = ROBOT_MODEL
        schema["grasp"] = {
            "location": "opposing side-wall midpoints at half box height",
            "pad_center_offsets_metres": [list(point) for point in GRIP_POINTS],
            "frame": "upper_box local frame; origin at bottom centre",
            "approach_tilt_from_vertical_radians": GRIP_APPROACH_TILT,
        }
        native = self.environment
        schema["initialization"] = {
            "distribution": "seeded uniform XY/yaw proposals, conditioned on minimum radial/angular offsets and both arms' existing workspace",
            "low": list(native.initial_offset_low),
            "high": list(native.initial_offset_high),
            "min_distance": native.minimum_initial_distance,
            "units": "metres",
            "minimum_distance_applies_to": "randomized resets only",
            "yaw_distribution": "uniform proposals about box centre, rejecting small absolute yaw and poses outside the two-arm workspace",
            "yaw_low_radians": native.initial_yaw_low,
            "yaw_high_radians": native.initial_yaw_high,
            "minimum_absolute_yaw_radians": native.minimum_initial_yaw,
            "yaw_positive_direction": "clockwise viewed from above",
            "workspace_rejection": True,
            "yaw_limit_radians": native.yaw_limit,
            "current_initial_yaw_radians": native.initial_yaw,
            "current_yaw_randomized": bool(native.initial_yaw_randomized),
            "box_z_metres": 0.2,
            "explicit_offset_priority": True,
            "fixed_offset_metres": native.fixed_offset.tolist(),
            "fixed_yaw_radians": native.fixed_yaw,
            "current_initial_offset_metres": native.initial_offset.tolist(),
            "current_initial_randomized": bool(native.initial_randomized),
            "current_offset_scope": "Snapshot when this schema is read; each episode's actual initial offset is its first observation.state XY and native info.initial_offset.",
            "seed_rule": "reset(seed) reproduces the same sample; new saved episodes use new seeds and discarded attempts reuse their seed",
        }
        schema["episode_lifecycle"] = {
            "lift_metres": 0.02, "xy_success_metres": 0.005,
            "yaw_success_radians": 0.01, "goal_yaw_radians": 0.0,
            "xy_success_metric": "radial_xy_distance = sqrt((upper.x - lower.x)^2 + (upper.y - lower.y)^2)",
            "xy_success_comparison": "<=",
            "landing_height_tolerance_metres": 0.0002, "success_hold_seconds": 0.5,
            "height_semantics": {
                "lifted_height": "Actual upper-box Z minus its original resting Z, in metres; a rise, not an error.",
                "lift_error": "Signed lifted_height minus 0.02 m; zero means the actual lift is 20 mm.",
                "landing_height_error": "Signed actual upper-box Z minus its original resting Z, in metres.",
                "height_error": "Signed actual upper-box Z minus the active stage target: original Z + 0.02 m in lifting/control, original Z in lowering/settling/complete.",
            },
            "phases": ["lifting", "control", "lowering", "settling", "complete", "failed"],
        }
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
        from .native.model import UPPER_REST_Z
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
            "observation.box_target_pose": (["target.x", "target.y", "target.z", "target.yaw"], ["m"] * 3 + ["rad"]),
            "observation.yaw": (["box.yaw"], ["rad"]),
        }
        for key, (names, units) in fields.items():
            schema["features"][key] = {"dtype": "float32", "shape": [len(names)], "names": names}
            schema["feature_semantics"][key] = {"units": units, "description": "Native pre-action world/robot state; all modalities are recorded."}
        schema["feature_semantics"]["observation.yaw"]["description"] = "Box-centred yaw in radians, clockwise positive and counterclockwise negative when viewed from above. Equal to observation.state[51]; native world quaternions remain right-handed."
        schema["feature_semantics"]["observation.box_target_pose"]["description"] = "Commanded box-centre XYZ in metres and clockwise-positive yaw in radians, before this row's action."
        schema["features"]["observation.n_contacts"] = {"dtype": "int64", "shape": [1], "names": ["count"]}
        schema["feature_semantics"]["observation.n_contacts"] = {"description": "MuJoCo contact count at this observation time; not contact-force magnitude."}
        schema["goal_pose"] = [0, 0, UPPER_REST_Z, 1, 0, 0, 0]
        schema["features"]["observation.phase"] = {"dtype": "int64", "shape": [1], "names": ["phase"]}
        schema["feature_semantics"]["observation.phase"] = {
            "description": "Episode stage before action: 0 lifting, 1 control, 2 lowering, 3 settling, 4 complete, 5 failed."
        }
        return schema

    def recording_observation(self, *, include_images: bool = True) -> dict[str, np.ndarray]:
        from .native.model import UPPER_REST_Z
        observation = self.lerobot_observation(include_images=include_images)
        state = self._latest_observation
        observation.update({
            "observation.keypoints": self.get_keypoints().reshape(-1),
            "observation.agent_pos": state[37:43].copy(),
            "observation.environment_state": state.copy(),
            "observation.velocity": state[7:13].copy(),
            "observation.goal_pose": np.asarray([0, 0, UPPER_REST_Z, 1, 0, 0, 0], dtype=np.float32),
            "observation.joint_positions": state[13:25].copy(),
            "observation.joint_velocities": state[25:37].copy(),
            "observation.tip_targets": state[43:49].copy(),
            "observation.contact_forces": state[52:54].copy(),
            "observation.box_target_pose": np.asarray([*self.environment.box_target_center,
                                                        self.environment.box_target_yaw], dtype=np.float32),
            "observation.yaw": state[51:52].copy(),
            "observation.n_contacts": np.asarray([self.environment.data.ncon], dtype=np.int64),
            "observation.phase": np.asarray([
                ["lifting", "control", "lowering", "settling", "complete", "failed"].index(self.environment.phase)
            ], dtype=np.int64),
        })
        return observation

