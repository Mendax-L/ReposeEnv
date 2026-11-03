"""Native field metadata and the LeRobot schema selected by each environment YAML."""
from __future__ import annotations

import math
from typing import Any, Sequence
from .config import load_environment_config

CAMERA_KEY = "observation.images.front"
CAMERA2_KEY = "observation.images.camera2"
SUPPORTED_ENVIRONMENTS = ("pusht", "alignbox3d")


def _vector(names):
    return {"dtype": "float32", "shape": [len(names)], "names": list(names)}


def get_environment_schema(name: str, image_shape: Sequence[int] | None = None) -> dict[str, Any]:
    """Generate flattened dataset features and named interface metadata together."""
    name = name.lower()
    if name not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(f"Unknown environment {name!r}; choose from {SUPPORTED_ENVIRONMENTS}.")
    config = load_environment_config(name)
    shape = list(image_shape if image_shape is not None else
                 ((512, 512, 3) if name == "alignbox3d" else (96, 96, 3)))
    if len(shape) != 3 or any(type(size) is not int or size <= 0 for size in shape) or shape[2] != 3:
        raise ValueError("image_shape must contain positive integer [height, width, 3] dimensions.")
    available, semantics = {}, {}
    if name == "alignbox3d":
        for side in ("left", "right"):
            for suffix, components, unit in (
                ("end_effector_pos", list("xyz"), "m"),
                ("end_effector_rot6d", [f"column{c}.{a}" for c in (0, 1) for a in "xyz"], "unitless"),
                ("end_effector_wxyz", list("wxyz"), "unitless"),
                ("end_effector_rotm", [f"{r}{c}" for r in range(3) for c in range(3)], "unitless"),
                ("gripper", ["position"], "m"),
            ):
                key = f"robot.{side}_{suffix}"
                available[key] = _vector(components)
                semantics[key] = {"units": [unit] * len(components)}
                if suffix != "gripper":
                    semantics[key]["coordinate_frame"] = "world"
                else:
                    semantics[key]["measurement"] = "Measured driven-finger slide travel, not the gap between pads."
            for suffix, components, scale in (("xyz", list("xyz"), .001),
                                               ("rpy", ["roll", "pitch", "yaw"], math.radians(.5))):
                key = f"action.{side}_end_effector_delta_{suffix}"
                available[key] = _vector(components)
                semantics[key] = {"components": components, "units": ["normalized_delta"] * 3,
                                  "low": [-1.] * 3, "high": [1.] * 3, "scale_to_si": [scale] * 3}
                if suffix == "rpy":
                    semantics[key].update(
                        coordinate_frame="local end-effector", rotation_order=["roll", "pitch", "yaw"],
                        rotation_composition="R_next = R_previous @ Rz(yaw) @ Ry(pitch) @ Rx(roll)")
                else:
                    semantics[key]["coordinate_frame"] = "world"
        camera = "images.camera2"
        camera_shape = [max(2, 2 * round(shape[0] * .3)), shape[1], 3]
        layout = "configured_robot_fields/v2"
        action_semantics = "dual_ee_delta_xyz_local_roll_pitch_yaw/v2"
        termination = "Alignment and stable release are evaluated by the native simulator; time/manual limits are truncated."
    else:
        available["robot.agent_pos"] = _vector(["x", "y"])
        semantics["robot.agent_pos"] = {"units": ["workspace_pixel"] * 2}
        available["action"] = _vector(["agent.target.x", "agent.target.y"])
        semantics["action"] = {"units": ["workspace_pixel"] * 2, "low": [0.] * 2,
                               "high": [512.] * 2, "coordinate_frame": "workspace XY; X right, Y down"}
        camera, camera_shape = "images.front", shape
        layout, action_semantics = "configured_robot_fields/v2", "absolute_agent_xy"
        termination = "terminated = coverage > 0.95; collector limits are truncated."
    available[camera] = {"dtype": "video", "shape": camera_shape,
                         "names": ["height", "width", "channels"]}
    semantics[camera] = {"array_dtype": "uint8", "array_layout": "HWC", "color_space": "RGB"}
    available["instruction"] = {"dtype": "string", "shape": [1], "names": ["instruction"]}
    semantics["instruction"] = {"description": "Natural-language task condition synchronized with this observation."}

    selected = config["action"] + [field for fields in config["observation"].values() for field in fields]
    if len(set(selected)) != len(selected) or any(field not in available for field in selected):
        raise ValueError("Interface YAML contains duplicated or unavailable fields.")
    interface_features = {field: available[field] for field in selected}
    state_names, state_units, action_names, action_units, low, high = [], [], [], [], [], []
    for key in config["observation"]["robot_state"]:
        state_names.extend(f"{key}.{component}" for component in available[key]["names"])
        state_units.extend(semantics[key]["units"])
    action_groups = {}
    for key in config["action"]:
        start = len(action_names)
        action_names.extend(f"{key}.{component}" for component in available[key]["names"])
        action_groups[key] = list(range(start, len(action_names)))
        action_units.extend(semantics[key]["units"])
        low.extend(semantics[key]["low"])
        high.extend(semantics[key]["high"])
    features = {"observation.state": _vector(state_names), "action": _vector(action_names)}
    feature_semantics = {
        "observation.state": {"description": "Measured robot fields concatenated in YAML order.", "units": state_units},
        "action": {"description": action_semantics, "units": action_units, "low": low, "high": high},
    }
    for key in config["observation"]["env"]:
        if key.startswith("images."):
            features[f"observation.{key}"] = available[key]
            feature_semantics[f"observation.{key}"] = semantics[key]
    return {
        "environment": name, "robot_type": name, "fps": config["fps"],
        "observation_layout": layout, "action_semantics": action_semantics,
        "features": features, "feature_semantics": feature_semantics, "termination": termination,
        "interface": config, "interface_features": interface_features,
        "action_groups": action_groups,
        "interface_semantics": {field: semantics[field] for field in selected},
    }
