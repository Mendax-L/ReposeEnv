"""Opt-in visual variants; original collision geoms and dynamics are retained.

The extra geoms have zero mass and no collision bits. Edge/face ribs and
container contents are visual structures, not new contact surfaces.
All changes are episode-constant and recorded outside observation.state.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import itertools
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.environment import PushEnvironment
from .env import AlignBox3DEnv
from .native.env import StackedBoxesEnv
from .native.model import BOX_SIZE, TABLE_HEIGHT, UPPER_REST_Z, build_scene
from .stage_labels import LANGUAGE_FEATURES, language_metadata, stage_observation


RIBBED_LEVELS = (0, 1, 2)
BOX_FORMS = ("plain", "edge_ribbed", "face_ribbed")
CONTENT_TYPES = ("none", "cartons", "cylinders", "mixed")
CONTENT_AMOUNTS = ("none", "few", "many")
# Rows follow CONTENT_TYPES; columns follow CONTENT_AMOUNTS.
CONTENT_COUNTS = (
    (0, 0, 0),
    (0, 3, 7),
    (0, 4, 8),
    (0, 4, 8),
)
CONTENT_STATES = (
    ("none", 0),
    ("cartons", 1), ("cartons", 2),
    ("cylinders", 1), ("cylinders", 2),
    ("mixed", 1), ("mixed", 2),
)
VISUAL_MODES = ("baseline", "color", "ribbing", "contents", "camera", "lighting", "all")
# Muted industrial plastic colours, not exact commercial/RAL specifications.
COLOR_STYLE = "industrial_muted/v1"
RENDER_STYLE = "industrial_workcell_shadow/v2"
BOX_PALETTE = (
    ("graphite", (.29, .31, .32)),
    ("slate_gray", (.47, .50, .51)),
    ("steel_blue", (.32, .42, .50)),
    ("olive_gray", (.45, .49, .40)),
    ("sand_beige", (.69, .64, .54)),
    ("warm_gray", (.63, .61, .57)),
    ("dusty_blue", (.48, .56, .61)),
    ("off_white", (.77, .76, .71)),
)
TABLE_PALETTE = (
    ("steel_gray", (.64, .66, .67)),
    ("warm_gray", (.63, .61, .58)),
    ("medium_gray", (.48, .50, .51)),
    ("light_gray", (.77, .77, .75)),
)
CONTENT_PALETTES = (
    ("kraft", ((.59, .43, .27, 1.), (.76, .64, .43, 1.), (.31, .26, .20, 1.))),
    ("cool_industrial", ((.25, .31, .34, 1.), (.43, .54, .60, 1.), (.67, .70, .68, 1.))),
    ("safety", ((.34, .36, .36, 1.), (.82, .43, .13, 1.), (.68, .68, .62, 1.))),
)
# Exact discrete combinations reserved for evaluation. Continuous camera and
# light values are derived from visual_seed; collectors reject these signatures.
HELDOUT_VISUAL_CONFIGS = (
    {"name": "plain_white_cylinders", "visual_seed": 900001, "ribbed_level": 0,
     "upper_color_id": 0, "lower_color_id": 7, "table_color_id": 1,
     "content_type_id": 2, "content_amount_id": 2, "content_palette_id": 2},
    {"name": "edge_blue_cartons", "visual_seed": 900002, "ribbed_level": 1,
     "upper_color_id": 6, "lower_color_id": 4, "table_color_id": 3,
     "content_type_id": 1, "content_amount_id": 1, "content_palette_id": 0},
    {"name": "face_olive_mixed", "visual_seed": 900003, "ribbed_level": 2,
     "upper_color_id": 3, "lower_color_id": 2, "table_color_id": 2,
     "content_type_id": 3, "content_amount_id": 2, "content_palette_id": 1},
    {"name": "plain_warm_mixed", "visual_seed": 900004, "ribbed_level": 0,
     "upper_color_id": 5, "lower_color_id": 2, "table_color_id": 0,
     "content_type_id": 3, "content_amount_id": 1, "content_palette_id": 2},
    {"name": "edge_graphite_cylinders", "visual_seed": 900005, "ribbed_level": 1,
     "upper_color_id": 7, "lower_color_id": 0, "table_color_id": 2,
     "content_type_id": 2, "content_amount_id": 1, "content_palette_id": 1},
    {"name": "face_beige_none", "visual_seed": 900006, "ribbed_level": 2,
     "upper_color_id": 4, "lower_color_id": 1, "table_color_id": 1,
     "content_type_id": 0, "content_amount_id": 0, "content_palette_id": 0},
)
HOLDOUT_SIGNATURE_FIELDS = ("ribbed_level", "upper_color_id", "lower_color_id", "table_color_id",
                            "content_type_id", "content_amount_id", "content_palette_id")
FEATURES = {
    "expert_stage_id": ("int64", [1]),
    "grasp_points_box_m": ("float32", [2, 3]),
    "grasp_angle_vectors_box_rad": ("float32", [2, 3]),
    "pair_id": ("int64", [1]),
    "pose_seed": ("int64", [1]),
    "visual_seed": ("int64", [1]),
    "visual_mode_id": ("int64", [1]),
    "box_form_id": ("int64", [1]),
    "ribbed_level": ("int64", [1]),
    "content_type_id": ("int64", [1]),
    "content_amount_id": ("int64", [1]),
    "content_count": ("int64", [1]),
    "content_palette_id": ("int64", [1]),
    "factors_enabled": ("bool", [5]),
    "color_style_id": ("int64", [1]),
    "upper_color_id": ("int64", [1]),
    "lower_color_id": ("int64", [1]),
    "table_color_id": ("int64", [1]),
    "structure_color_scale": ("float32", [1]),
    "upper_rgba": ("float32", [4]),
    "lower_rgba": ("float32", [4]),
    "table_rgba": ("float32", [4]),
    "content_rgba": ("float32", [3, 4]),
    "camera_position_m": ("float32", [3]),
    "camera_delta_m": ("float32", [3]),
    "camera_quaternion_wxyz": ("float32", [4]),
    "camera_delta_rpy_deg": ("float32", [3]),
    "light_scale": ("float32", [2]),
    "initial_offset_m": ("float32", [2]),
    "initial_yaw_rad": ("float32", [1]),
}


@dataclass(frozen=True)
class VisualConfig:
    camera_position_jitter_m: float = .01
    camera_rotation_jitter_deg: float = 1.
    light_min: float = .8
    light_max: float = 1.2

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if not np.isfinite(values).all():
            raise ValueError("Visual limits must be finite.")
        if not 0 <= self.camera_position_jitter_m <= .02:
            raise ValueError("Camera position jitter must be in [0, .02] metres.")
        if not 0 <= self.camera_rotation_jitter_deg <= 2:
            raise ValueError("Camera rotation jitter must be in [0, 2] degrees.")
        if not .5 <= self.light_min <= self.light_max <= 1.5:
            raise ValueError("Light range must lie within [.5, 1.5].")


def visual_scene() -> str:
    """Compile all visual variants once; switch alpha without recompilation."""
    root = ET.fromstring(build_scene())
    for box in ("upper", "lower"):
        body = root.find(f"./worldbody/body[@name='{box}_box']")
        original = list(body.findall("geom"))
        for level in RIBBED_LEVELS[1:]:
            serial = 0

            def add(pos, size, accent=False):
                nonlocal serial
                ET.SubElement(body, "geom", {
                    "name": f"visual_{box}_ribbed{level}_{'accent' if accent else 'shell'}_{serial}",
                    "type": "box", "pos": " ".join(map(str, pos)),
                    "size": " ".join(map(str, size)), "mass": "0",
                    "contype": "0", "conaffinity": "0", "group": "2",
                    "material": f"{box}_box_material",
                    "rgba": "0.5 0.5 0.5 0",
                })
                serial += 1

            for geom in original:
                pos = np.fromstring(geom.get("pos"), sep=" ")
                size = np.fromstring(geom.get("size"), sep=" ")
                if level == 2 and "wall_y_" in geom.get("name"):
                    # Recess only the broad visible faces. The X walls below
                    # the grippers retain the original outer envelope.
                    sign = np.sign(pos[1])
                    add([0, sign * .1435, pos[2]], [.12, .0015, size[2]])
                    for x in np.linspace(-.114, .114, 7):
                        add([x, sign * .1475, pos[2]], [.006, .0025, size[2]], True)
                    for z in (.013, .115):
                        add([0, sign * .1475, z], [.12, .0025, .005], True)
                else:
                    add(pos, size)
            if level == 1:
                # Edge-only structure: no ribs across either broad face.
                for sign in (-1, 1):
                    for x in (-.114, .114):
                        add([x, sign * .1475, .064], [.006, .0025, .056], True)
                    for z in (.013, .115):
                        add([0, sign * .1475, z], [.12, .0025, .005], True)

    container = root.find("./worldbody/body[@name='upper_box']")
    content_shapes = {
        "cartons": (
            ("box", (-.072, -.082, .031), (.026, .031, .027), 0),
            ("box", (0., -.084, .027), (.025, .029, .023), 1),
            ("box", (.072, -.080, .030), (.026, .032, .026), 2),
            ("box", (-.074, -.005, .026), (.024, .029, .022), 1),
            ("box", (0., -.002, .032), (.027, .030, .028), 0),
            ("box", (.074, .004, .027), (.024, .028, .023), 2),
            ("box", (-.025, .078, .029), (.032, .030, .025), 0),
        ),
        "cylinders": (
            ("cylinder", (-.084, -.064, .030), (.019, .026), 0),
            ("cylinder", (-.028, -.066, .028), (.018, .024), 1),
            ("cylinder", (.028, -.063, .032), (.020, .028), 2),
            ("cylinder", (.084, -.065, .029), (.019, .025), 0),
            ("cylinder", (-.084, .064, .029), (.019, .025), 1),
            ("cylinder", (-.028, .066, .031), (.020, .027), 2),
            ("cylinder", (.028, .063, .028), (.018, .024), 0),
            ("cylinder", (.084, .065, .030), (.019, .026), 1),
        ),
        "mixed": (
            ("box", (-.073, -.072, .029), (.025, .030, .025), 0),
            ("cylinder", (-.022, -.073, .029), (.018, .025), 1),
            ("box", (.035, -.072, .026), (.024, .029, .022), 2),
            ("cylinder", (.086, -.071, .031), (.019, .027), 0),
            ("cylinder", (-.084, .066, .028), (.018, .024), 2),
            ("box", (-.031, .067, .027), (.024, .029, .023), 1),
            ("cylinder", (.027, .066, .030), (.019, .026), 0),
            ("box", (.083, .066, .028), (.024, .030, .024), 2),
        ),
    }
    for content_type, shapes in content_shapes.items():
        for serial, (kind, pos, size, palette_slot) in enumerate(shapes):
            ET.SubElement(container, "geom", {
                "name": f"visual_content_{content_type}_slot{palette_slot}_{serial}",
                "type": kind, "pos": " ".join(map(str, pos)),
                "size": " ".join(map(str, size)), "mass": "0",
                "contype": "0", "conaffinity": "0", "group": "2",
                "rgba": "0 0 0 0",
            })
    return ET.tostring(root, encoding="unicode")


def _seed_rng(seed, stream):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seeds must be integers in [0, 2**63).")
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _swatch(rng, palette):
    """Small shared brightness changes keep each named material recognisable."""
    index = int(rng.integers(len(palette)))
    rgb = np.array(palette[index][1]) * rng.uniform(.95, 1.05) + rng.uniform(-.006, .006, 3)
    return index, np.r_[np.clip(rgb, 0., 1.), 1.]


def industrial_colors(rng, color_ids=None):
    """Neutral work surface and distinct but plausible plastic box colours."""
    if color_ids is not None:
        if len(color_ids) != 3:
            raise ValueError("color_ids must contain upper, lower and table palette IDs.")
        upper_id, lower_id, table_id = (int(value) for value in color_ids)
        if not (0 <= upper_id < len(BOX_PALETTE) and 0 <= lower_id < len(BOX_PALETTE)
                and 0 <= table_id < len(TABLE_PALETTE)):
            raise ValueError("A color_ids entry is outside its palette.")
        colors = tuple(np.r_[palette[index][1], 1.] for palette, index in
                       ((BOX_PALETTE, upper_id), (BOX_PALETTE, lower_id),
                        (TABLE_PALETTE, table_id)))
        return colors, (upper_id, lower_id, table_id)
    upper_id, upper = _swatch(rng, BOX_PALETTE)
    luminance = np.array([.2126, .7152, .0722])
    for _ in range(256):
        lower_id, lower = _swatch(rng, BOX_PALETTE)
        difference = upper[:3] - lower[:3]
        if np.linalg.norm(difference) >= .16 and abs(difference @ luminance) >= .055:
            break
    else:
        raise RuntimeError("Cannot sample distinct industrial box colours.")
    for _ in range(256):
        table_id, table = _swatch(rng, TABLE_PALETTE)
        if min(np.linalg.norm(table[:3] - box[:3]) for box in (upper, lower)) >= .12:
            break
    else:
        raise RuntimeError("Cannot sample a neutral table with visible box edges.")
    return (upper, lower, table), (upper_id, lower_id, table_id)


def visual_signature(metadata):
    """Return the discrete fields used to keep evaluation visuals out of training."""
    return tuple(int(metadata[field]) for field in HOLDOUT_SIGNATURE_FIELDS)


_HELDOUT_SIGNATURES = frozenset(visual_signature(row) for row in HELDOUT_VISUAL_CONFIGS)


def is_heldout_configuration(metadata):
    return visual_signature(metadata) in _HELDOUT_SIGNATURES


def _delta_quat(rpy):
    result = np.array([1., 0., 0., 0.])
    for axis, angle in enumerate(np.radians(rpy)):
        q = np.array([math.cos(angle / 2), 0., 0., 0.])
        q[axis + 1] = math.sin(angle / 2)
        out = np.empty(4)
        mujoco.mju_mulQuat(out, result, q)
        result = out
    return result


def camera_contains_boxes(native, margin=.025):
    """Frame upper-box corners and lower-box alignment walls over the workspace.

    This checks framing, not occlusion by robot parts. The overview camera is
    deliberately unchanged; only closeup/camera2 is randomized.
    """
    model = native.model
    cam = model.camera("closeup").id
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, model.cam_quat[cam])
    angle = native.yaw_limit
    x = (BOX_SIZE[0] * math.cos(angle) + BOX_SIZE[1] * math.sin(angle)) / 2 + .055
    y = (BOX_SIZE[1] * math.cos(angle) + BOX_SIZE[0] * math.sin(angle)) / 2 + .040
    points = list(itertools.product((-x, x), (-y, y),
                                   (UPPER_REST_Z, UPPER_REST_Z + .02 + BOX_SIZE[2])))
    # The original waist camera already crops the bottom of the lower box.
    # Preserve its visible upper half and alignment rim, not the unseen base.
    points += list(itertools.product((-.12, .12), (-.15, .15),
                                    (TABLE_HEIGHT + BOX_SIZE[2] / 2, UPPER_REST_Z)))
    local = (np.asarray(points) - model.cam_pos[cam]) @ matrix.reshape(3, 3)
    depth = -local[:, 2]
    height = max(2, 2 * round(native.render_size * .3))
    half_y = depth * math.tan(math.radians(model.cam_fovy[cam]) / 2)
    half_x = half_y * native.render_size / height
    return bool(np.all(depth > 0) and np.all(np.abs(local[:, 0]) <= half_x * (1 - margin))
                and np.all(np.abs(local[:, 1]) <= half_y * (1 - margin)))


class DiverseAlignBox3DEnv(AlignBox3DEnv):
    """Opt-in adapter retaining the old state/action and recording contracts."""

    def __init__(self, render_size=512, visual_config=None):
        self.visual_config = visual_config or VisualConfig()
        PushEnvironment.__init__(self, StackedBoxesEnv(render_size=render_size, scene_xml=visual_scene()))
        model = self.environment.model
        self._base = {key: getattr(model, key).copy() for key in
                      ("geom_rgba", "mat_rgba", "cam_pos", "cam_quat", "light_diffuse")}
        self.visual_metadata = {}
        self.collection_metadata = {}
        self._ribbed_geoms = {level: [] for level in RIBBED_LEVELS}
        self._content_geoms = {name: [] for name in CONTENT_TYPES}
        self._collision_boxes = []
        for i in range(model.ngeom):
            name = model.geom(i).name
            if name.startswith("visual_content_"):
                self._content_geoms[name.split("_")[2]].append(i)
            elif name.startswith(("visual_upper_ribbed", "visual_lower_ribbed")):
                self._ribbed_geoms[int(name.split("_")[2].removeprefix("ribbed"))].append(i)
            elif name.startswith(("upper_", "lower_")):
                self._collision_boxes.append(i)

    def reset(self, seed, *, options=None):
        options = dict(options or {})
        visual_seed = int(options.pop("visual_seed", seed))
        mode = options.pop("visual_mode", "all")
        if mode == "form":
            mode = "ribbing"
        legacy_form = options.pop("box_form", None)
        ribbed_level = options.pop("ribbed_level", None)
        content_type = options.pop("content_type", None)
        content_amount = options.pop("content_amount", None)
        color_ids = options.pop("color_ids", None)
        content_palette_id = options.pop("content_palette_id", None)
        aliases = {"plain": 0, "edge_ribbed": 1, "face_ribbed": 2, "ribbed": 2}
        if legacy_form is not None:
            if legacy_form not in aliases:
                raise ValueError("box_form must be plain, edge_ribbed or face_ribbed.")
            legacy_level = aliases[legacy_form]
            if ribbed_level is not None and int(ribbed_level) != legacy_level:
                raise ValueError("box_form and ribbed_level disagree.")
            ribbed_level = legacy_level
        if mode not in VISUAL_MODES:
            raise ValueError("Unknown visual mode.")
        if ribbed_level is not None and (type(ribbed_level) is not int or ribbed_level not in RIBBED_LEVELS):
            raise ValueError("ribbed_level must be 0, 1 or 2.")
        if content_type == "empty":
            content_type = "none"
        if content_type is not None and content_type not in CONTENT_TYPES:
            raise ValueError(f"content_type must be one of {CONTENT_TYPES}.")
        if isinstance(content_amount, str):
            if content_amount not in CONTENT_AMOUNTS:
                raise ValueError(f"content_amount must be one of {CONTENT_AMOUNTS}.")
            content_amount = CONTENT_AMOUNTS.index(content_amount)
        if content_amount is not None and (type(content_amount) is not int
                                           or not 0 <= content_amount < len(CONTENT_AMOUNTS)):
            raise ValueError("content_amount must be none/few/many or 0/1/2.")
        if content_type == "none" and content_amount not in (None, 0):
            raise ValueError("content_type='none' requires content_amount='none'.")
        if content_palette_id is not None and (type(content_palette_id) is not int
                                               or not 0 <= content_palette_id < len(CONTENT_PALETTES)):
            raise ValueError("content_palette_id is outside the content palette.")
        observation = super().reset(seed, options=options)
        self._apply_visuals(visual_seed, mode, ribbed_level, content_type, content_amount,
                            color_ids, content_palette_id)
        self.collection_metadata = {
            "expert_stage_id": -1,
            "grasp_points_box_m": [[0., 0., 0.], [0., 0., 0.]],
            "grasp_angle_vectors_box_rad": [[0., 0., 0.], [0., 0., 0.]],
            "pair_id": int(seed), "pose_seed": int(seed),
            "initial_offset_m": self.environment.initial_offset.tolist(),
            "initial_yaw_rad": float(self.environment.initial_yaw),
        }
        self.language_stage_id = -1
        return observation

    def _apply_visuals(self, seed, mode, ribbed_level, content_type, content_amount,
                       color_ids, content_palette_id):
        model = self.environment.model
        for key, values in self._base.items():
            getattr(model, key)[:] = values
        enabled = [mode in (name, "all") for name in
                   ("color", "ribbing", "contents", "camera", "lighting")]
        # Separate streams give identical factor values in single-factor and
        # combined runs, independent of the other enabled factors.
        rng = _seed_rng(seed, 11)
        upper = self._base["mat_rgba"][model.material("upper_box_material").id].copy()
        lower = self._base["mat_rgba"][model.material("lower_box_material").id].copy()
        table = self._base["mat_rgba"][model.material("table_material").id].copy()
        selected_color_ids = (-1, -1, -1)
        if enabled[0]:
            (upper, lower, table), selected_color_ids = industrial_colors(rng, color_ids)
            for name, color in (("upper_box_material", upper), ("lower_box_material", lower),
                                ("table_material", table)):
                model.mat_rgba[model.material(name).id] = color
        level = ((ribbed_level if ribbed_level is not None else
                  int(_seed_rng(seed, 12).integers(len(RIBBED_LEVELS)))) if enabled[1] else 0)
        # Moulded ribs share the shell colour with a subtle shade.
        structure_color_scale = .90 if enabled[0] else .65
        if level > 0:
            model.geom_rgba[self._collision_boxes, 3] = 0
            for i in self._ribbed_geoms[level]:
                name = model.geom(i).name
                color = upper.copy() if name.startswith("visual_upper_") else lower.copy()
                if "_accent_" in name:
                    color[:3] *= structure_color_scale
                model.geom_rgba[i] = color

        if enabled[2]:
            if content_amount is None:
                # Existing callers that specify only a type keep the old full
                # content layout; fully random calls sample all three levels.
                amount_id = (0 if content_type == "none" else 2) if content_type is not None else int(
                    _seed_rng(seed, 17).integers(len(CONTENT_AMOUNTS)))
            else:
                amount_id = content_amount
            if amount_id == 0:
                selected_content = "none"
            else:
                selectable = CONTENT_TYPES[1:]
                selected_content = (content_type if content_type not in (None, "none") else
                                    selectable[int(_seed_rng(seed, 15).integers(len(selectable)))])
        else:
            selected_content, amount_id = "none", 0
        palette_id = ((content_palette_id if content_palette_id is not None else
                       int(_seed_rng(seed, 16).integers(len(CONTENT_PALETTES))))
                      if enabled[2] else -1)
        content_rgba = np.zeros((3, 4), dtype=np.float64)
        if enabled[2]:
            content_rgba[:] = np.asarray(CONTENT_PALETTES[palette_id][1], dtype=np.float64)
            content_id = CONTENT_TYPES.index(selected_content)
            visible_count = CONTENT_COUNTS[content_id][amount_id]
            for i in self._content_geoms[selected_content][:visible_count]:
                slot = int(model.geom(i).name.split("_slot")[1].split("_")[0])
                model.geom_rgba[i] = content_rgba[slot]
        cam = model.camera("closeup").id
        delta = np.zeros(3)
        rpy = np.zeros(3)
        rejected = 0
        if enabled[3]:
            rng = _seed_rng(seed, 13)
            config = self.visual_config
            for rejected in range(256):
                delta = rng.uniform(-config.camera_position_jitter_m, config.camera_position_jitter_m, 3)
                rpy = rng.uniform(-config.camera_rotation_jitter_deg, config.camera_rotation_jitter_deg, 3)
                model.cam_pos[cam] = self._base["cam_pos"][cam] + delta
                mujoco.mju_mulQuat(model.cam_quat[cam], self._base["cam_quat"][cam], _delta_quat(rpy))
                if camera_contains_boxes(self.environment):
                    break
            else:
                raise RuntimeError("Cannot frame both boxes with the requested camera jitter.")
        scale = np.ones(2)
        if enabled[4]:
            scale = _seed_rng(seed, 14).uniform(self.visual_config.light_min, self.visual_config.light_max, 2)
            for name, factor in zip(("key", "fill"), scale):
                lid = model.light(name).id
                model.light_diffuse[lid] = self._base["light_diffuse"][lid] * factor
        mujoco.mj_forward(model, self.environment.data)
        self.visual_metadata = {
            "visual_seed": seed, "visual_mode_id": VISUAL_MODES.index(mode),
            "box_form_id": level, "ribbed_level": level,
            "content_type_id": CONTENT_TYPES.index(selected_content),
            "content_amount_id": amount_id,
            "content_count": CONTENT_COUNTS[CONTENT_TYPES.index(selected_content)][amount_id],
            "content_palette_id": palette_id, "factors_enabled": enabled,
            "color_style_id": int(enabled[0]),
            "upper_color_id": selected_color_ids[0], "lower_color_id": selected_color_ids[1],
            "table_color_id": selected_color_ids[2],
            "structure_color_scale": structure_color_scale,
            "upper_rgba": upper.tolist(), "lower_rgba": lower.tolist(), "table_rgba": table.tolist(),
            "content_rgba": content_rgba.tolist(),
            "camera_position_m": model.cam_pos[cam].tolist(), "camera_delta_m": delta.tolist(),
            "camera_quaternion_wxyz": model.cam_quat[cam].tolist(), "camera_delta_rpy_deg": rpy.tolist(),
            "light_scale": scale.tolist(), "camera_proposal_rejections": rejected,
        }

    def recording_schema(self):
        schema = super().recording_schema()
        for key, (dtype, shape) in FEATURES.items():
            schema["features"][f"collection.{key}"] = {"dtype": dtype, "shape": shape, "names": None}
            schema["feature_semantics"][f"collection.{key}"] = {
                "description": "Episode-constant ablation metadata; NOT a policy observation. See meta/diversity.json."
            }
        schema["visual_diversity"] = {
            "config": asdict(self.visual_config), "box_forms": list(BOX_FORMS),
            "ribbed_levels": list(RIBBED_LEVELS), "content_types": list(CONTENT_TYPES),
            "content_amounts": list(CONTENT_AMOUNTS),
            "render_style": RENDER_STYLE,
            "physics": "Original five collision geoms, mass, grip points and outer box dimensions retained.",
            "structures": "Edge and face ribs plus box contents are zero-mass, collision-disabled visuals.",
        }
        for key in ("expert_stage_id", "grasp_points_box_m", "grasp_angle_vectors_box_rad"):
            schema["feature_semantics"][f"collection.{key}"]["description"] = (
                "Grasp expert metadata, excluded from policy inputs: internal stage ID or sampled box-local contact points/rotation vectors. Stage may change each frame; -1 means no grasp expert is running.")
        schema["features"].update(copy.deepcopy(LANGUAGE_FEATURES))
        schema["language_stages"] = language_metadata()
        for key in LANGUAGE_FEATURES:
            schema["feature_semantics"][key] = {"description": "Current grasp-stage instruction before the action; see meta/language.json. Explicit policy language condition, separate from observation.state."}
        return schema

    def recording_observation(self, *, include_images=True):
        observation = super().recording_observation(include_images=include_images)
        metadata = {**self.visual_metadata, **self.collection_metadata}
        for key, (dtype, shape) in FEATURES.items():
            observation[f"collection.{key}"] = np.asarray(metadata[key], dtype=dtype).reshape(shape)
        observation.update(stage_observation(self.language_stage_id))
        return observation

    def episode_metadata(self):
        metadata = copy.deepcopy({**self.visual_metadata, **self.collection_metadata})
        metadata.update(visual_mode=VISUAL_MODES[metadata["visual_mode_id"]],
                        box_form=BOX_FORMS[metadata["box_form_id"]],
                        content_type=CONTENT_TYPES[metadata["content_type_id"]],
                        content_amount=CONTENT_AMOUNTS[metadata["content_amount_id"]],
                        render_style=RENDER_STYLE)
        metadata["color_style"] = COLOR_STYLE if metadata["color_style_id"] else "original"
        metadata["content_palette"] = (CONTENT_PALETTES[metadata["content_palette_id"]][0]
                                       if metadata["content_palette_id"] >= 0 else "none")
        for part, palette in (("upper", BOX_PALETTE), ("lower", BOX_PALETTE), ("table", TABLE_PALETTE)):
            index = metadata[f"{part}_color_id"]
            metadata[f"{part}_color_name"] = palette[index][0] if index >= 0 else "original"
        return metadata
