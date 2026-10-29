"""Two real-size ARX L5 arms with independent end-effector control in MuJoCo.

All distances are metres and all joint angles are radians.  Each box consists
of exactly five rigid box geoms: one floor and four walls.  The upper box is a
free body moved by physical contact; the lower box is fixed. Reset starts
with open grippers clear of the box.
The vendored Google DeepMind MuJoCo
Menagerie ARX L5 description supplies the original kinematic tree, inertia,
joint limits, actuator torque limits, visual meshes, and collision geometry.
Ideal gravity compensation (body gravcomp=1) is applied to both arms.
This is a simulation assumption, not a model of the hardware controller.
"""

from __future__ import annotations

import math
import copy
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET


BOX_SIZE = (0.24, 0.30, 0.12)
WALL_THICKNESS = 0.008
UPPER_MASS = 0.65
TABLE_HEIGHT = 0.08
UPPER_REST_Z = TABLE_HEIGHT + BOX_SIZE[2]
ROBOT_MODEL = "Bimanual ARX L5"
ARX_L5_DIRECTORY = Path(__file__).resolve().parent / "assets" / "arx_l5"
# Pad half-thickness is 2 mm and its fixed centre offset is 1.898 mm.
# Centres at ±6 mm therefore leave an 8 mm clear gap at reset.
FINGER_OPENING = WALL_THICKNESS / 2 + .002 - .001898
# Use the centres of the original distal finger pads as the holding frame.
TIP_LOCAL_POSITION = (.13657, 0., -.00024363)
TIP_SITE_QUAT = (math.sqrt(.5), 0., math.sqrt(.5), 0.)
ARM_NAMES = ("inner", "outer")
# Grip halfway up both side walls, at their fore/aft midpoint. The fingers
# descend 10 degrees from vertical, keeping the original palms above the rim
# and both arms inside their original joint limits. Closing remains world X.
GRIP_POINTS = (
    (-(BOX_SIZE[0] - WALL_THICKNESS) / 2, 0., BOX_SIZE[2] / 2),
    ((BOX_SIZE[0] - WALL_THICKNESS) / 2, 0., BOX_SIZE[2] / 2),
)
GRIP_APPROACH_TILT = math.radians(10.)
_s, _c = math.sin(GRIP_APPROACH_TILT), math.cos(GRIP_APPROACH_TILT)
TIP_ROTATIONS = {
    arm: ((0., -1., 0.), (-_c, 0., _s), (-_s, 0., -_c))
    for arm in ARM_NAMES
}
_qs = math.sin(GRIP_APPROACH_TILT / 2) * math.sqrt(.5)
_qc = math.cos(GRIP_APPROACH_TILT / 2) * math.sqrt(.5)
GRIP_QUATS = {
    arm: (_qs, -_qc, _qc, _qs) for arm in ARM_NAMES
}
ARM_BASES = {
    "inner": (-0.27, -0.35, TABLE_HEIGHT),
    "outer": (0.27, -0.35, TABLE_HEIGHT),
}
# Preserve the original keyframe as provenance. These seeds place the pad
# centres at the side-wall midpoints of the aligned, resting upper box.
SOURCE_ARM_HOME = (0., .251, .314, 0., 0., 0.)
ARM_HOME = {
    "inner": (1.12484747, 1.87783265, 1.86675280, -1.40195542, -.07496382, -.44002113),
    "outer": (2.01673489, 1.87783273, 1.86675292, -1.40195569, .07496879, .44000942),
}
HOME_QPOS = ARM_HOME
JOINT_LIMITS = (
    (-3.14, 3.14), (0., 3.14), (0., 3.14),
    (-1.7, 1.7), (-1.7, 1.7), (-3.14, 3.14),
)


def _fmt(values: Sequence[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _box_geoms(prefix: str, material: str, mass: float) -> str:
    """Partition the shell without overlapping solids; masses sum to mass."""
    width, depth, height = BOX_SIZE
    wall = WALL_THICKNESS
    wall_height = height - wall
    geometries = (
        ("floor", (0, 0, wall / 2), (width / 2, depth / 2, wall / 2)),
        (
            "wall_x_neg",
            (-(width - wall) / 2, 0, (height + wall) / 2),
            (wall / 2, (depth - 2 * wall) / 2, wall_height / 2),
        ),
        (
            "wall_x_pos",
            ((width - wall) / 2, 0, (height + wall) / 2),
            (wall / 2, (depth - 2 * wall) / 2, wall_height / 2),
        ),
        (
            "wall_y_neg",
            (0, -(depth - wall) / 2, (height + wall) / 2),
            (width / 2, wall / 2, wall_height / 2),
        ),
        (
            "wall_y_pos",
            (0, (depth - wall) / 2, (height + wall) / 2),
            (width / 2, wall / 2, wall_height / 2),
        ),
    )
    total_volume = sum(8 * size[0] * size[1] * size[2] for _, _, size in geometries)
    lines = []
    for name, pos, size in geometries:
        geom_mass = mass * 8 * size[0] * size[1] * size[2] / total_volume
        lines.append(
            f'<geom name="{prefix}_{name}" type="box" pos="{_fmt(pos)}" '
            f'size="{_fmt(size)}" mass="{geom_mass:.10g}" '
            f'material="{material}" class="box_contact"/>'
        )
    return "\n".join(lines)


def _arm_sections(name: str) -> dict[str, str]:
    """Namespace the original ARX L5, including its two real sliding fingers."""
    source = ARX_L5_DIRECTORY / "arx_l5.xml"
    if not source.is_file():
        raise FileNotFoundError(f"ARX L5 model asset is missing: {source}")
    arm = copy.deepcopy(ET.parse(source).getroot())
    aliases = {f"joint{i}": f"{name}_j{i}" for i in range(1, 9)}
    aliases.update({"base_link": f"{name}_base", "link7": f"{name}_finger_left",
                    "link8": f"{name}_finger_right"})
    def prefixed(value):
        return aliases.get(value, f"{name}_{value}")
    for mesh in arm.findall("./asset/mesh"):
        mesh.set("file", (ARX_L5_DIRECTORY / "assets" / mesh.attrib["file"]).as_posix())
    ref_attributes = {
        "class", "childclass", "material", "mesh", "texture", "joint",
        "joint1", "joint2", "body1", "body2", "geom1", "geom2", "site",
        "site1", "site2", "target", "tendon", "cranksite", "slidersite",
    }
    for element in arm.iter():
        if "name" in element.attrib:
            element.set("name", prefixed(element.attrib["name"]))
        for attribute in ref_attributes:
            if attribute in element.attrib:
                element.set(attribute, prefixed(element.attrib[attribute]))
        if element.tag == "body":
            element.set("gravcomp", "1")
    for index, actuator in enumerate(arm.findall("./actuator/*"), 1):
        actuator.set("name", f"{name}_j{index}_act" if index <= 6 else f"{name}_gripper_act")
        if index <= 6:
            # Position-loop tuning for the shared 650 g payload; retain the
            # source actuator's original 100 N m torque saturation.
            actuator.set("kp", "80")
            actuator.set("kv", "5")
        else:
            # Metre-scale finger slides need enough position-loop stiffness
            # to hold the 650 g payload; retain the original +/-10 N limit.
            actuator.set("kp", "2500")
            actuator.set("kv", "40")
    collision_defaults = arm.find(f"./default/default/default[@class='{name}_collision']/geom")
    assert collision_defaults is not None
    collision_defaults.set("contype", "4")
    collision_defaults.set("conaffinity", "7")
    base = arm.find("./worldbody/body")
    assert base is not None
    base.set("pos", _fmt(ARM_BASES[name]))
    base.set("childclass", f"{name}_arx_l5")
    for index, geom in enumerate(base.iter("geom")):
        kind = "visual" if geom.get("class") == f"{name}_visual" else "collision"
        geom.set("name", f"{name}_{kind}_{index:02d}")
    for side, compatibility_name in (("left", "probe"), ("right", "tip_geom")):
        finger = next(body for body in base.iter("body") if body.get("name") == f"{name}_finger_{side}")
        geoms = finger.findall("geom")
        geoms[0].set("name", f"{name}_finger_{side}_visual")
        for index, geom in enumerate(geoms[1:]):
            geom.set("name", f"{name}_{compatibility_name}" if index == 0 else f"{name}_finger_{side}_base")
            geom.set("contype", "2")
            geom.set("conaffinity", "7")
            geom.set("friction", ".8 .005 .0001")
    wrist = next(body for body in base.iter("body") if body.get("name") == f"{name}_link6")
    ET.SubElement(wrist, "site", {
        "name": f"{name}_tip", "pos": _fmt(TIP_LOCAL_POSITION), "quat": _fmt(TIP_SITE_QUAT),
        "size": ".003", "rgba": "1 .95 .35 1", "group": "4",
    })
    ET.SubElement(wrist, "site", {"name": f"{name}_wrist", "size": ".003", "group": "4"})
    sections = {}
    for tag in ("default", "asset", "actuator", "contact", "equality"):
        element = arm.find(tag)
        sections[tag] = "" if element is None else "\n".join(ET.tostring(child, encoding="unicode") for child in element)
    sections["body"] = ET.tostring(base, encoding="unicode")
    return sections


def build_scene(offset: Sequence[float] = (0.035, -0.02), yaw: float = 0.0) -> str:
    """Return MJCF with two ARX L5 arms and their original mesh assets.

    ``offset`` gives the upper box's initial world x/y displacement relative to
    the lower box. ``yaw`` rotates about its central vertical axis, positive
    clockwise when viewed from above (opposite the native quaternion Z angle).
    Initial arm joint angles must be set to ``ARM_HOME`` after model loading;
    MuJoCo's default zero pose is deliberately not encoded as joint references.
    """
    if len(offset) != 2 or not all(math.isfinite(float(value)) for value in (*offset, yaw)):
        raise ValueError("offset must contain two finite coordinates and yaw must be finite")
    x, y = (float(value) for value in offset)
    upper_quat = _fmt((math.cos(yaw / 2), 0, 0, -math.sin(yaw / 2)))
    lower = _box_geoms("lower", "lower_box_material", 2.0)
    upper = _box_geoms("upper", "upper_box_material", UPPER_MASS)
    sections = [_arm_sections(name) for name in ARM_NAMES]
    arms = "\n".join(section["body"] for section in sections)
    arm_defaults = "\n".join(section["default"] for section in sections)
    arm_assets = "\n".join(section["asset"] for section in sections)
    actuators = "\n".join(section["actuator"] for section in sections)
    contacts = "\n".join(section["contact"] for section in sections)
    arm_equalities = "\n".join(section["equality"] for section in sections)
    return f"""<mujoco model="dual_arm_open_box_alignment">
  <compiler angle="radian" autolimits="true"/>
  <option timestep=".002" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" iterations="80" tolerance="1e-10" cone="elliptic"/>
  <size nconmax="300" njmax="1500"/>
  <visual>
    <global offwidth="1280" offheight="960" azimuth="90" elevation="-75"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient=".18 .18 .18" diffuse=".25 .25 .25" specular=".06 .06 .06"/>
    <map znear=".01" zfar="20"/>
  </visual>
  <statistic center="0 0 .20" extent="1.3"/>
  <default>
    <geom solref=".012 1" solimp=".94 .99 .001" margin="0"/>
    <default class="box_contact">
      <geom contype="1" conaffinity="7" condim="4" friction=".15 .005 .0001"/>
    </default>
    <default class="probe_contact">
      <geom contype="2" conaffinity="7" condim="4" priority="1" friction=".05 .005 .0001"/>
    </default>
    {arm_defaults}
  </default>
  <asset>
    {arm_assets}
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".48 .50 .52"
             rgb2=".30 .32 .34" width="512" height="3072"/>
    <texture name="floor_checker" type="2d" builtin="checker" rgb1=".19 .22 .28"
             rgb2=".23 .26 .32" width="512" height="512"/>
    <material name="floor_material" texture="floor_checker" texrepeat="4 4" reflectance=".04"/>
    <material name="table_material" rgba=".70 .74 .79 1" specular=".25" shininess=".25"/>
    <material name="lower_box_material" rgba=".18 .42 .62 1" specular=".16" shininess=".22"/>
    <material name="upper_box_material" rgba=".95 .55 .17 1" specular=".16" shininess=".22"/>
    <material name="inner_paint" rgba=".10 .65 .67 1" specular=".5" shininess=".5"/>
    <material name="outer_paint" rgba=".38 .45 .83 1" specular=".5" shininess=".5"/>
    <material name="dark_metal" rgba=".13 .16 .21 1" specular=".5" shininess=".5"/>
    <material name="joint_metal" rgba=".58 .64 .70 1" specular=".65" shininess=".65"/>
    <material name="probe_metal" rgba=".79 .84 .9 1" specular=".8" shininess=".8"/>
    <material name="tip_rubber" rgba=".14 .17 .20 1" specular=".1" shininess=".1"/>
  </asset>
  <worldbody>
    <light name="key" pos="-1.3 -1.1 2.6" dir=".4 .3 -1" diffuse=".58 .57 .55"
           directional="true" castshadow="true"/>
    <light name="fill" pos="1.5 .5 1.8" dir="-.6 -.2 -1" diffuse=".27 .28 .30" castshadow="false"/>
    <camera name="overview" pos="0 -.15 2.4" xyaxes="1 0 0 0 1 0" fovy="26"/>
    <camera name="top" pos="0 0 1.7" xyaxes="1 0 0 0 1 0" fovy="48"/>
    <!-- Look down about 17 degrees: keep the two front walls and alignment edges
         below the wrists, while preserving visible depth and yaw offsets. -->
    <camera name="closeup" pos="0 -1.5 .6719237886" xyaxes="1 0 0 0 0.2884742946 0.9574876403" fovy="12.8"/>
    <geom name="ground" type="plane" pos="0 0 -.64" size="3 3 .1"
          material="floor_material" friction=".8 .01 .001"/>
    <geom name="table_top" type="box" pos="0 0 .04" size=".63 .63 .04"
          material="table_material" friction=".7 .01 .001"/>
    <geom name="table_leg_1" type="box" pos="-.53 -.53 -.32" size=".035 .035 .32" material="dark_metal"/>
    <geom name="table_leg_2" type="box" pos="-.53 .53 -.32" size=".035 .035 .32" material="dark_metal"/>
    <geom name="table_leg_3" type="box" pos=".53 -.53 -.32" size=".035 .035 .32" material="dark_metal"/>
    <geom name="table_leg_4" type="box" pos=".53 .53 -.32" size=".035 .035 .32" material="dark_metal"/>
    <body name="lower_box" pos="0 0 {TABLE_HEIGHT}">
      {lower}
      <site name="target_center" pos="0 0 {BOX_SIZE[2]}" size=".003" rgba=".2 1 .4 .5" group="4"/>
    </body>
    <body name="upper_box" pos="{x:.10g} {y:.10g} {UPPER_REST_Z:.10g}" quat="{upper_quat}">
      <freejoint name="upper_free"/>
      {upper}
      <site name="upper_center" pos="0 0 0" size=".003" rgba="1 .2 .2 .5" group="4"/>
      <site name="inner_grip" pos="{_fmt(GRIP_POINTS[0])}" quat="{_fmt(GRIP_QUATS['inner'])}" size=".003" rgba=".2 1 .4 1" group="4"/>
      <site name="outer_grip" pos="{_fmt(GRIP_POINTS[1])}" quat="{_fmt(GRIP_QUATS['outer'])}" size=".003" rgba=".2 1 .4 1" group="4"/>
    </body>
    {arms}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
  <contact>
    {contacts}
  </contact>
  <equality>
    {arm_equalities}
  </equality>
</mujoco>
"""
