"""LeRobot feature definitions for ReposeEnv, without simulator or torch imports.

The PushT adapter retains its existing 20-value policy state. Coordinates use
the native 512 by 512 workspace, not image pixels.
The 3-D adapter retains the original 54 fields and appends four gripper states and eight absolute tool-quaternion components.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


CAMERA_KEY = "observation.images.front"
CAMERA2_KEY = "observation.images.camera2"
SUPPORTED_ENVIRONMENTS = ("pusht", "alignbox3d")
ALIGNBOX3D_OBSERVATION_FIELDS = (
    "box_pose", "box_velocity", "joint_positions", "joint_velocities",
    "tip_positions", "tip_targets", "alignment_error", "contact_forces",
    "gripper_positions", "gripper_targets", "tip_quaternions",
)


def _vector(names: list[str]) -> dict[str, Any]:
    return {"dtype": "float32", "shape": [len(names)], "names": names}


def _alignbox3d_state() -> tuple[list[str], list[str]]:
    names = [f"box.position.{axis}" for axis in "xyz"]
    units = ["m"] * 3
    names += [f"box.quaternion.{axis}" for axis in "wxyz"]
    units += ["unitless"] * 4
    names += [f"box.linear_velocity.{axis}" for axis in "xyz"]
    units += ["m/s"] * 3
    names += [f"box.angular_velocity.{axis}" for axis in "xyz"]
    units += ["rad/s"] * 3
    for quantity, unit in (("position", "rad"), ("velocity", "rad/s")):
        names += [f"{arm}.joint_{joint}.{quantity}" for arm in ("inner", "outer") for joint in range(1, 7)]
        units += [unit] * 12
    for quantity in ("position", "target"):
        names += [f"{arm}.tip_{quantity}.{axis}" for arm in ("inner", "outer") for axis in "xyz"]
        units += ["m"] * 6
    names += ["alignment_error.x", "alignment_error.y", "alignment_error.yaw"]
    units += ["m", "m", "rad"]
    names += ["inner.contact_force", "outer.contact_force"]
    units += ["N"] * 2
    for quantity in ("position", "target"):
        names += [f"{arm}.gripper.{quantity}" for arm in ("inner", "outer")]
        units += ["m"] * 2
    names += [f"{arm}.tip_quaternion.{axis}" for arm in ("inner", "outer") for axis in "wxyz"]
    units += ["unitless"] * 8
    return names, units


def get_environment_schema(
    name: str, image_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return a fresh JSON-serializable schema and separate physical semantics.

    ``features`` can be passed to a LeRobot dataset writer. ``feature_semantics``
    is additional ReposeEnv metadata; it does not add nonstandard feature dtypes.
    Images supplied by the environment are RGB uint8 HWC arrays and stored as
    videos. FPS is the control/data sampling frequency, not the physics rate.
    """
    name = name.lower()
    if name not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"Unknown environment {name!r}; choose from {SUPPORTED_ENVIRONMENTS}.")
    shape = list(image_shape if image_shape is not None else ((512, 512, 3) if name == "alignbox3d" else (96, 96, 3)))
    if len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape) or shape[2] != 3:
        raise ValueError("image_shape must contain positive integer [height, width, 3] dimensions.")

    if name == "alignbox3d":
        state_names, state_units = _alignbox3d_state()
        action_names = [field for arm in ("inner", "outer")
                        for field in ([f"{arm}.tip.delta.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")]
                                      + [f"{arm}.gripper.opening"])]
        layout = "stacked_box_state_yaw_cw_grippers_tool_quaternions/v4"
        action_semantics = "dual_ee_world_xyz_rotvec_delta_1mm_0.5deg_gripper_abs_0_1/v4"
        state_description = (
            "Native fields flattened in order: " + ", ".join(ALIGNBOX3D_OBSERVATION_FIELDS) + ". "
            "box_pose is upper-box free-joint qpos: world XYZ then quaternion WXYZ. "
            "box_velocity follows native MuJoCo free-joint qvel order: linear XYZ, angular XYZ. "
            "Joint order is inner joints 1-6 then outer joints 1-6. Tip coordinates are world metres. "
            "Alignment error is upper-box world X/Y and clockwise-positive yaw relative to the fixed lower box. "
            "Native quaternions and angular velocities retain the right-handed world convention; clockwise yaw is the negative of the quaternion's world-Z angle. "
            "The XY success metric is radial XY distance sqrt(dx^2 + dy^2), with an inclusive 0.005 m limit. "
            "Contact forces are normal gripper-to-upper-box forces averaged over 10 physics substeps. State indices 54:56 are measured driven-finger slide positions and 56:58 are their targets in metres, inner then outer; they are not measured pad gaps. State indices 58:66 are inner then outer measured end-effector WXYZ quaternions in the shared world frame. Measured XYZ positions remain at 37:43. Both tools use the same origin and axes, never separate arm-base frames."
        )
        action_metadata = {
            "description": "Direct independent end-effector pose increments: inner dx/dy/dz/rx/ry/rz/opening, then outer dx/dy/dz/rx/ry/rz/opening. Each opening is an absolute normalized target: 0 closed, 1 fully open, mapped to the driven finger slide range 0..0.044 m. It is not a delta, force or measured pad gap. Zero in both gripper channels commands closing; use the current opening ratios to hold them. Translation is in world axes; rotation is a right-handed world-frame rotation vector, applied about each tool origin by left multiplication. Each translation component scales to 0.001 m and rotation-vector component to 0.5 degrees per step. No automatic lift, height correction or landing. Workspace/IK may shorten or reject a pose increment; the box moves only through physical contact. Grippers start open, and both tools park at world XY (-0.20,-0.10) and (0.20,-0.10) metres. Their initial Z is chosen so the lowest visible/collision finger point is at least 0.020 m above the rotated box highest point; flat-box tool-site Z is 0.362 m. Initial tool orientation is the original grasp orientation rotated 20 degrees about world X.",
            "units": (["normalized_delta"] * 6 + ["normalized_opening"]) * 2,
            "low": ([-1.0] * 6 + [0.0]) * 2,
            "high": [1.0] * 14,
            "scale_to_metres": 0.001,
            "rotation_scale_to_radians": math.radians(.5),
            "scale_to_si": ([0.001] * 3 + [math.radians(.5)] * 3 + [0.044]) * 2,
            "gripper_action_indices": [6, 13],
            "gripper_target_range_metres": [0.0, 0.044],
            "gripper_command_type": "absolute normalized driven-finger slide position",
            "rotation_representation": "world-frame rotation vector (axis times angle), not Euler angles",
            "rotation_convention": "right_hand_positive; +rz is counterclockwise viewed from above",
            "rotation_pivot": "each end-effector origin independently",
            "coordinate_frame": "world",
        }
        termination = "Reset directly into control with open, detached tools and the lowest rotated box-bottom corner on the Z=0.200 m rim plane; gravity can naturally settle the initial roll/pitch. Success requires radial XY distance <= 0.005 m, absolute yaw <= 3 degrees, absolute height error from 0.200 m < 0.0002 m and tilt < 0.02 rad, with speed norm < 0.015, continuously for 0.5 s. Manual/time-limit completion without success is truncated; no lift or landing stages."

    else:
        points = [f"object.keypoint_{index}" for index in range(9)]
        state_description = "Nine T-block keypoints in the native PymunkKeypointManager order (default sampling seed 0), followed by the external agent XY. Visibility masks are omitted; the adapter uses visibility 1.0."
        agent_names = ["agent.x", "agent.y"]
        action_names = ["agent.target.x", "agent.target.y"]
        layout = "object9_xy_agent_xy/v1"
        action_semantics = "absolute_agent_xy"
        action_description = "Absolute agent target XY in the native 512 by 512 workspace, followed by the native PD controller."
        state_names = [f"{point}.{axis}" for point in points for axis in "xy"] + agent_names
        state_units = ["workspace_pixel"] * len(state_names)
        action_metadata = {
            "description": action_description + " Coordinates are not normalized to image resolution.",
            "units": ["workspace_pixel"] * len(action_names),
            "low": [0.0] * len(action_names),
            "high": [512.0] * len(action_names),
            "coordinate_frame": "workspace image coordinates; X right, Y down",
        }
        termination = "terminated = coverage > 0.95; no native time-limit truncation. Collection limits must be marked truncated by the collector."

    schema = {
        "environment": name,
        "robot_type": name,
        "fps": 50 if name == "alignbox3d" else 10,
        "observation_layout": layout,
        "action_semantics": action_semantics,
        "features": {
            "observation.state": _vector(state_names),
            "action": _vector(action_names),
            CAMERA_KEY: {"dtype": "video", "shape": shape, "names": ["height", "width", "channels"]},
        },
        "feature_semantics": {
            "observation.state": {"description": state_description, "units": state_units},
            "action": action_metadata,
            CAMERA_KEY: {
                "description": (
                    "Native camera1 overview RGB: the complete target region, both end effectors and both arm bases; 'front' retains the common stream key."
                    if name == "alignbox3d" else
                    "Native top-down RGB rendering without action/UI markers; 'front' is the common stream key."
                ),
                "array_dtype": "uint8", "array_layout": "HWC", "color_space": "RGB",
            },
        },
        "termination": termination,
    }
    if name == "alignbox3d":
        schema["features"][CAMERA2_KEY] = {
            "dtype": "video", "shape": [max(2, 2 * round(shape[0] * 0.3)), shape[1], 3],
            "names": ["height", "width", "channels"],
        }
        schema["feature_semantics"][CAMERA2_KEY] = {
            "description": "Native camera2 elevated oblique closeup RGB: upper-box sidewalls, lower target region and both end effectors, with both arm bases outside the frame.",
            "array_dtype": "uint8", "array_layout": "HWC", "color_space": "RGB",
        }
    return schema
