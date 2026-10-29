"""AlignBox3D collection metadata, independent of command-line entry points."""
from __future__ import annotations

from dataclasses import asdict

from .diversity import (
    BOX_FORMS, BOX_PALETTE, COLOR_STYLE, CONTENT_AMOUNTS, CONTENT_PALETTES,
    CONTENT_TYPES, FEATURES, HELDOUT_VISUAL_CONFIGS, HOLDOUT_SIGNATURE_FIELDS,
    RENDER_STYLE, RIBBED_LEVELS, TABLE_PALETTE, VISUAL_MODES,
)


def metadata_dictionary(config):
    return {
        "version": 5, "config": asdict(config), "render_style": RENDER_STYLE,
        "color_style_id": {0: "original", 1: COLOR_STYLE},
        "box_color_id": {index: {"name": name, "base_rgb": list(rgb)} for index, (name, rgb) in enumerate(BOX_PALETTE)},
        "table_color_id": {index: {"name": name, "base_rgb": list(rgb)} for index, (name, rgb) in enumerate(TABLE_PALETTE)},
        "color_sampling": "Named muted industrial palettes, brightness x[.95,1.05] and RGB +/- .006. Colour ID -1 means original fixed material. Actual RGBA values are stored per episode/frame.",
        "content_palette_id": {index: {"name": name, "rgba": [list(color) for color in colors]}
                               for index, (name, colors) in enumerate(CONTENT_PALETTES)},
        "structure_color_scale": "Multiplier applied to ribs: .90 in industrial colour mode, .65 with original colours.",
        "box_form_id": dict(enumerate(BOX_FORMS)), "visual_mode_id": dict(enumerate(VISUAL_MODES)),
        "ribbed_level": {level: BOX_FORMS[level] for level in RIBBED_LEVELS},
        "content_type_id": dict(enumerate(CONTENT_TYPES)),
        "content_amount_id": dict(enumerate(CONTENT_AMOUNTS)),
        "factors_enabled_order": ["color", "ribbing", "contents", "camera", "lighting"],
        "features": {f"collection.{key}": {"dtype": dtype, "shape": shape} for key, (dtype, shape) in FEATURES.items()},
        "units": {"*_m": "metres", "*_rad": "radians, clockwise-positive yaw",
                  "*_deg": "degrees", "*_mm_s": "mm/s", "*_deg_s": "degrees/s",
                  "*_rgba": "renderer RGBA values in [0,1]", "light_scale": "relative to original lights"},
        "pair_id": "Same seed and pair index across visual modes: identical physical initialization, motion and factor random streams. Group all equal pair_ids when splitting train/validation.",
        "camera": "Only camera2/closeup changes. Position deltas in world XYZ; rotation delta is local X then Y then Z quaternion composition. Both deltas are sampled once per episode. FOV unchanged.",
        "geometry": "Nominal dimensions 240x300x120 mm. Original collision shell, inertia and gripper geometry unchanged. Ribs and contents are visual-only and collision-disabled.",
        "heldout_signature_fields": list(HOLDOUT_SIGNATURE_FIELDS),
        "heldout_visual_configs": list(HELDOUT_VISUAL_CONFIGS),
        "heldout_policy": "Collectors reject every reserved discrete signature; these configurations are evaluation-only.",
        "expert_labels": "Only successful contact-grasp demonstrations are saved; per-frame stage instructions are defined in meta/language.json.",
        "policy_inputs": "collection.* are experimental metadata, excluded from observation.state and action; existing image/state models ignore them.",
    }
