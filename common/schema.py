"""LeRobot feature definitions for pushenv, without simulator or torch imports.

The two-dimensional adapters retain their existing policy states (PushT: 20,
PushBox2D: 12). Coordinates use the native 512 by 512 workspace, not image pixels.
The 3-D adapter retains all 54 native state values in its established field order.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


CAMERA_KEY = "observation.images.front"
CAMERA2_KEY = "observation.images.camera2"
SUPPORTED_ENVIRONMENTS = ("pusht", "pushbox2d", "pushbox3d")
BOX3D_OBSERVATION_FIELDS = (
    "box_pose", "box_velocity", "joint_positions", "joint_velocities",
    "tip_positions", "tip_targets", "alignment_error", "contact_forces",
)


def _vector(names: list[str]) -> dict[str, Any]:
    return {"dtype": "float32", "shape": [len(names)], "names": names}


def _box3d_state() -> tuple[list[str], list[str]]:
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
    return names, units


def get_environment_schema(
    name: str, image_shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return a fresh JSON-serializable schema and separate physical semantics.

    ``features`` can be passed to a LeRobot dataset writer. ``feature_semantics``
    is additional pushenv metadata; it does not add nonstandard feature dtypes.
    Images supplied by the environment are RGB uint8 HWC arrays and stored as
    videos. FPS is the control/data sampling frequency, not the physics rate.
    """
    name = name.lower()
    if name not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"Unknown environment {name!r}; choose from {SUPPORTED_ENVIRONMENTS}.")
    shape = list(image_shape if image_shape is not None else ((512, 512, 3) if name == "pushbox3d" else (96, 96, 3)))
    if len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape) or shape[2] != 3:
        raise ValueError("image_shape must contain positive integer [height, width, 3] dimensions.")

    if name == "pushbox3d":
        state_names, state_units = _box3d_state()
        action_names = [f"box.delta.{axis}" for axis in ("x", "y", "yaw")]
        layout = "stacked_box_state_yaw_cw/v2"
        action_semantics = "box_center_xy_yaw_cw_delta_1mm_0.5deg_auto_lift_20mm/v2"
        state_description = (
            "Native fields flattened in order: " + ", ".join(BOX3D_OBSERVATION_FIELDS) + ". "
            "box_pose is upper-box free-joint qpos: world XYZ then quaternion WXYZ. "
            "box_velocity follows native MuJoCo free-joint qvel order: linear XYZ, angular XYZ. "
            "Joint order is inner joints 1-6 then outer joints 1-6. Tip coordinates are world metres. "
            "Alignment error is upper-box world X/Y and clockwise-positive yaw relative to the fixed lower box. "
            "Native quaternions and angular velocities retain the right-handed world convention; clockwise yaw is the negative of the quaternion's world-Z angle. "
            "The XY success metric is radial XY distance sqrt(dx^2 + dy^2), with an inclusive 0.005 m limit. "
            "Contact forces are normal gripper-to-upper-box forces averaged over 10 physics substeps."
        )
        action_metadata = {
            "description": "Box-centred world XY translation and clockwise-positive yaw about the central vertical axis, viewed from above. One unit is 0.001 m on x/y or 0.5 degrees on yaw per control step. Both grip positions and tool orientations follow the rigid box pose. Automatic lift/lower/settle ignores all three action channels.",
            "units": ["normalized_delta"] * 3,
            "low": [-1.0] * 3,
            "high": [1.0] * 3,
            "scale_to_metres": 0.001,
            "yaw_scale_to_radians": math.radians(.5),
            "scale_to_si": [0.001, 0.001, math.radians(.5)],
            "rotation_axis": "vertical Z through box centre; positive yaw is clockwise viewed from above (right-hand axis -Z)",
            "yaw_convention": "clockwise_positive_viewed_from_above",
            "rotation_pivot": "upper-box centre, never a gripper or the world origin",
            "coordinate_frame": "world",
        }
        termination = "Auto lift 0.02 m, box-centred XY/yaw control, auto lower, then verify stable landed alignment for 0.5 s. Success requires radial XY distance sqrt(dx^2 + dy^2) <= 0.005 m, absolute landing height error < 0.0002 m, yaw < 0.01 rad and tilt < 0.02 rad; manual or time-limit completion without success is truncated."
    else:
        if name == "pusht":
            points = [f"object.keypoint_{index}" for index in range(9)]
            state_description = "Nine T-block keypoints in the native PymunkKeypointManager order (default sampling seed 0), followed by the external agent XY. Visibility masks are omitted; the adapter uses visibility 1.0."
            agent_names = ["agent.x", "agent.y"]
            action_names = ["agent.target.x", "agent.target.y"]
            layout = "object9_xy_agent_xy/v1"
            action_semantics = "absolute_agent_xy"
            action_description = "Absolute agent target XY in the native 512 by 512 workspace, followed by the native PD controller."
        else:
            points = [f"object.corner_{index}" for index in range(4)]
            state_description = "Four outer box corners in get_keypoints()/Pymunk vertex order, then the left and right attached-grip XY, retaining the external_agent/internal_agent field names. Corners encode translation and rotation continuously without a wrapped angle field."
            agent_names = ["external_agent.x", "external_agent.y", "internal_agent.x", "internal_agent.y"]
            action_names = ["grip.target.x", "grip.target.y"]
            layout = "box4_xy_two_agents_xy/v1"
            action_semantics = "attached_pair_absolute_xy"
            action_description = "Absolute target XY for the midpoint of the two attached grips in the native 512 by 512 workspace; both grips move together."
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
        "fps": 50 if name == "pushbox3d" else 10,
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
                    "Native camera1 overview RGB: the complete target region, both attached grips and both arm bases; 'front' retains the common stream key."
                    if name == "pushbox3d" else
                    "Native top-down RGB rendering without action/UI markers; 'front' is the common stream key."
                ),
                "array_dtype": "uint8", "array_layout": "HWC", "color_space": "RGB",
            },
        },
        "termination": termination,
    }
    if name == "pushbox3d":
        schema["features"][CAMERA2_KEY] = {
            "dtype": "video", "shape": [max(2, 2 * round(shape[0] * 0.3)), shape[1], 3],
            "names": ["height", "width", "channels"],
        }
        schema["feature_semantics"][CAMERA2_KEY] = {
            "description": "Native camera2 elevated oblique closeup RGB: upper-box sidewalls, lower target region and both attached grips, with both arm bases outside the frame.",
            "array_dtype": "uint8", "array_layout": "HWC", "color_space": "RGB",
        }
    return schema
