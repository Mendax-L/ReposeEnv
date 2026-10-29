"""Direct bimanual Cartesian end-effector pose control."""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .model import (ARM_NAMES, ARM_HOME,
                    TIP_ROTATIONS, UPPER_REST_Z, build_scene)
from .kinematics import ArmIK
from ..geometry import box_rotation, quaternion


class StackedBoxesEnv(gym.Env):
    """14 actions: two independent tool-pose increments and gripper targets.

    Inner XYZ/rxryrz/opening precedes outer XYZ/rxryrz/opening. Reset parks both tools clear
    of the supported box with open fingers.
    No automatic grasp, lift, height correction or landing is performed.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    control_height = UPPER_REST_Z
    initial_tip_positions = ((-.20, -.10, .362), (.20, -.10, .362))
    initial_finger_clearance_m = .020
    initial_tool_tilt_radians = math.radians(20.)
    gripper_opening = .012
    xy_tolerance = .005  # Inclusive radial XY distance, not a per-axis bound.
    yaw_tolerance = math.radians(3.)  # Inclusive absolute yaw error.
    height_tolerance = .0002
    initial_offset_low = (-.040, -.030)
    initial_offset_high = (.040, .030)
    minimum_initial_distance = .020
    initial_yaw_low = math.radians(-8.)
    initial_yaw_high = math.radians(8.)
    minimum_initial_yaw = math.radians(3.)
    yaw_limit = math.radians(12.)
    rotation_action_scale = math.radians(.5)
    initial_tilt_range = math.radians(2.)
    initial_tilt_limit = math.radians(5.)

    def __init__(self, render_mode=None, offset=(.035, -.02), yaw=0.0,
                 max_episode_seconds=120.0, render_size=512, *, scene_xml=None):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.render_size = render_size
        self.fixed_offset = np.asarray(offset, dtype=float).copy()
        self.initial_offset = self.fixed_offset.copy()
        self.initial_randomized = False
        self.fixed_yaw = self.initial_yaw = float(yaw)
        self.initial_yaw_randomized = False
        self.model = mujoco.MjModel.from_xml_string(
            build_scene(offset, yaw) if scene_xml is None else scene_xml)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = int(max_episode_seconds / self.dt)
        self.action_space = spaces.Box(
            np.asarray([-1.] * 6 + [0.] + [-1.] * 6 + [0.], dtype=np.float32),
            np.ones(14, dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "box_pose": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float64),
            "box_velocity": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float64),
            "joint_positions": spaces.Box(-np.inf, np.inf, (12,), dtype=np.float64),
            "joint_velocities": spaces.Box(-np.inf, np.inf, (12,), dtype=np.float64),
            "tip_positions": spaces.Box(-np.inf, np.inf, (2, 3), dtype=np.float64),
            "tip_targets": spaces.Box(-np.inf, np.inf, (2, 3), dtype=np.float64),
            "alignment_error": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
            "contact_forces": spaces.Box(0., np.inf, (2,), dtype=np.float64),
            "gripper_positions": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float64),
            "gripper_targets": spaces.Box(0., np.inf, (2,), dtype=np.float64),
            "tip_quaternions": spaces.Box(-1., 1., (2, 4), dtype=np.float64),
        })
        self.arm_qids = []
        self.arm_dofs = []
        self.arm_aids = []
        self.arm_jids = []
        self.finger_qids = []
        self.gripper_aids = []
        for arm in ARM_NAMES:
            joints = [self.model.joint(f"{arm}_j{i}") for i in range(1, 7)]
            self.arm_qids.append(np.array([j.qposadr[0] for j in joints]))
            self.arm_dofs.append(np.array([j.dofadr[0] for j in joints]))
            self.arm_jids.append(np.array([j.id for j in joints]))
            self.arm_aids.append(np.array([self.model.actuator(f"{arm}_j{i}_act").id for i in range(1, 7)]))
            self.finger_qids.append(np.array([self.model.joint(f"{arm}_j{i}").qposadr[0]
                                             for i in (7, 8)]))
            self.gripper_aids.append(self.model.actuator(f"{arm}_gripper_act").id)
        self.gripper_limits = self.model.actuator_ctrlrange[self.gripper_aids].copy()
        self.gripper_targets = np.full(2, self.gripper_opening, dtype=float)
        self.qids = np.concatenate(self.arm_qids)
        self.dofs = np.concatenate(self.arm_dofs)
        self.box_qadr = self.model.joint("upper_free").qposadr[0]
        self.box_dadr = self.model.joint("upper_free").dofadr[0]
        self.box_id = self.model.body("upper_box").id
        self.upper_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.box_id))
        # Historical name retained: these are the actual finger collision boxes,
        # including both the distal pads and the proximal finger segments.
        self.probe_geoms = [set(np.flatnonzero(
            np.isin(self.model.geom_bodyid,
                    [self.model.body(f"{arm}_finger_{side}").id for side in ("left", "right")])
            & (self.model.geom_contype != 0))) for arm in ARM_NAMES]
        self.finger_geoms = [np.flatnonzero(np.isin(self.model.geom_bodyid,
            [self.model.body(f"{arm}_finger_{side}").id for side in ("left", "right")]))
            for arm in ARM_NAMES]
        self.robot_geoms = set()
        for gid in range(self.model.ngeom):
            body_name = self.model.body(int(self.model.geom_bodyid[gid])).name
            if body_name.startswith(tuple(f"{arm}_" for arm in ARM_NAMES)):
                self.robot_geoms.add(gid)
        self.tip_ids = [self.model.site(f"{arm}_tip").id for arm in ARM_NAMES]
        self.grip_ids = [self.model.site(f"{arm}_grip").id for arm in ARM_NAMES]
        self.rest_tip_rotations = np.asarray([TIP_ROTATIONS[arm] for arm in ARM_NAMES])
        # Cartesian command bounds include the collision-free parked poses;
        # the original robot joint limits remain enforced by numerical IK.
        self.workspace_low = np.array([-.22, -.15, .22])
        self.workspace_high = np.array([.22, .08, .42])
        self.ik_solvers = [ArmIK(self.model, self.tip_ids[i], self.arm_qids[i],
                                 self.arm_dofs[i], self.arm_jids[i],
                                 np.asarray(TIP_ROTATIONS[ARM_NAMES[i]]))
                           for i in range(2)]
        self.ik_q = np.array([ARM_HOME[arm] for arm in ARM_NAMES])
        self.tip_targets = np.zeros((2, 3))
        self.tip_target_rotations = np.zeros((2, 3, 3))
        self.contact_forces = np.zeros(2)
        self.peak_forces = np.zeros(2)
        self.had_contact = np.zeros(2, dtype=bool)
        self.unintended_contact_count = 0
        self.unintended_contact_peak = 0.
        self.unintended_contact_pairs = set()
        self._renderers = {}
        self.steps = self.stable_steps = 0
        self.ik_clipped = False
        self.reset()

    @property
    def tip_positions(self):
        return self.data.site_xpos[self.tip_ids].copy()

    @property
    def pose(self):
        p = self.data.xpos[self.box_id].copy()
        rot = self.data.xmat[self.box_id].reshape(3, 3)
        yaw = -math.atan2(rot[1, 0], rot[0, 0])
        return p, yaw

    @property
    def gripper_positions(self):
        """Measured driven-finger slide positions, inner then outer, in metres."""
        return self.data.qpos[[qids[0] for qids in self.finger_qids]].copy()

    @property
    def tip_rotations(self):
        return self.data.site_xmat[self.tip_ids].reshape(2, 3, 3).copy()

    @property
    def tip_quaternions(self):
        return np.asarray([quaternion(rotation) for rotation in self.tip_rotations])

    @property
    def tip_poses(self):
        """Both measured poses in the SAME world frame: XYZ then WXYZ."""
        return np.column_stack((self.tip_positions, self.tip_quaternions))

    @property
    def box_highest_z(self):
        from .model import BOX_SIZE
        corners = np.asarray([[x,y,z] for x in (-BOX_SIZE[0]/2, BOX_SIZE[0]/2)
                              for y in (-BOX_SIZE[1]/2, BOX_SIZE[1]/2) for z in (0., BOX_SIZE[2])])
        return float(np.max((corners @ self.data.xmat[self.box_id].reshape(3,3).T)[:,2])
                     + self.data.xpos[self.box_id,2])

    @property
    def finger_lowest_heights(self):
        """Lowest visible or collision finger point, inner then outer, in world Z."""
        heights = []
        for geoms in self.finger_geoms:
            lowest = np.inf
            for gid in geoms:
                row = self.data.geom_xmat[gid].reshape(3,3)[2]
                if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = self.model.geom_dataid[gid]
                    start = self.model.mesh_vertadr[mesh]
                    vertices = self.model.mesh_vert[start:start+self.model.mesh_vertnum[mesh]]
                    height = self.data.geom_xpos[gid,2] + np.min(vertices @ row)
                else:
                    height = self.data.geom_xpos[gid,2] - np.abs(row) @ self.model.geom_size[gid]
                lowest = min(lowest, float(height))
            heights.append(lowest)
        return np.asarray(heights)

    def hold_action(self):
        """Zero tool motion with each gripper retaining its commanded opening."""
        action = np.zeros((2, 7), dtype=np.float32)
        action[:, 6] = np.clip(
            (self.gripper_targets - self.gripper_limits[:, 0])
            / np.diff(self.gripper_limits, axis=1).ravel(), 0., 1.)
        return action.ravel()

    def set_targets(self, targets, rotations=None):
        """Solve the two requested tool poses without changing live body qpos."""
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (2, 3) or not np.all(np.isfinite(targets)):
            raise ValueError("targets must be a finite 2x3 array in world metres")
        rotations = self.tip_target_rotations.copy() if rotations is None else np.asarray(rotations, dtype=float)
        if (rotations.shape != (2, 3, 3) or not np.isfinite(rotations).all()
                or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-7)
                or not np.allclose(np.linalg.det(rotations), 1., atol=1e-7)):
            raise ValueError("rotations must contain two proper 3x3 rotation matrices")
        for i, solver in enumerate(self.ik_solvers):
            # Each arm reaches its own feasible target independently.
            for fraction in (1., .5, .25, .125, .0625):
                candidate = self.tip_targets[i] + fraction * (targets[i] - self.tip_targets[i])
                candidate_rotation = self._interpolate_rotation(self.tip_target_rotations[i], rotations[i], fraction)
                if np.any(candidate < self.workspace_low) or np.any(candidate > self.workspace_high):
                    continue
                solution = solver.solve(self.data.qpos, self.ik_q[i], candidate,
                                        rotation=candidate_rotation)
                if solver.converged:
                    self.tip_targets[i] = candidate
                    self.tip_target_rotations[i] = candidate_rotation
                    self.ik_q[i] = solution
                    self.ik_clipped |= fraction < 1.
                    break
            else:
                self.ik_clipped = True
    @staticmethod
    def _rotation_increment(vector):
        angle = float(np.linalg.norm(vector))
        if angle < 1e-12:
            return np.eye(3)
        x, y, z = vector / angle
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)

    @classmethod
    def _interpolate_rotation(cls, start, end, fraction):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (end @ start.T).ravel())
        if quat[0] < 0:
            quat *= -1
        length = np.linalg.norm(quat[1:])
        vector = quat[1:] * (2 * math.atan2(length, quat[0]) / max(length, 1e-12))
        return cls._rotation_increment(vector * fraction) @ start

    def _set_phase(self, phase):
        self.phase = phase
        self.phase_steps = 0
        self.stable_steps = 0

    def request_finish(self, reason="manual"):
        """End on the next step without inserting any robot motion."""
        if self.phase not in ("complete", "failed"):
            self.finish_requested = True
            self.finish_reason = str(reason)
        return self._info()

    def _advance_phase(self, info):
        if info["failure"]:
            self._set_phase("failed")
            return
        if self.phase != "control":
            return
        if self.finish_requested:
            self._set_phase("complete")
            return
        velocity = self.data.qvel[self.box_dadr:self.box_dadr + 6]
        aligned = (info["xy_error"] <= self.xy_tolerance
                   and info["yaw_error"] <= self.yaw_tolerance
                   and abs(info["height_error"]) < self.height_tolerance
                   and info["tilt"] < .02)
        ready = aligned and np.linalg.norm(velocity) < .015
        self.stable_steps = self.stable_steps + 1 if ready else 0
        if self.terminate_on_alignment and self.stable_steps >= round(.5 / self.dt):
            self.success = True
            self.finish_requested = True
            self.finish_reason = "aligned"
            self._set_phase("complete")

    def _sample_initial_offset(self):
        """Uniform XY proposal excluding the central 20 mm disk."""
        for _ in range(1024):
            offset = self.np_random.uniform(self.initial_offset_low, self.initial_offset_high)
            if np.linalg.norm(offset) >= self.minimum_initial_distance:
                return offset
        raise RuntimeError("Could not sample an initial XY offset outside the minimum-distance disk.")

    def _sample_initial_yaw(self):
        """Keep every randomized episode's clockwise-signed yaw visible."""
        for _ in range(1024):
            yaw = float(self.np_random.uniform(self.initial_yaw_low, self.initial_yaw_high))
            if abs(yaw) >= self.minimum_initial_yaw:
                return yaw
        raise RuntimeError("Could not sample an initial yaw outside the minimum-angle interval.")

    def reset(self, *, seed=None, options=None):
        """Seeded XY/yaw/roll/pitch; explicit options override each component.

        Reusing a seed reproduces its initial pose. ``randomize=False`` uses
        the constructor's fixed offset, preserving controlled-test entry points.
        """
        super().reset(seed=seed)
        options = options or {}
        self.terminate_on_alignment = options.get("terminate_on_alignment", True)
        if type(self.terminate_on_alignment) is not bool:
            raise ValueError("terminate_on_alignment must be a bool")
        randomized = "offset" not in options and bool(options.get("randomize", True))
        random_yaw = bool(options.get("randomize", True)) and "yaw" not in options
        random_tilt = bool(options.get("randomize_tilt", randomized and random_yaw))
        roll = float(options.get("roll", self.np_random.uniform(-self.initial_tilt_range, self.initial_tilt_range) if random_tilt else 0.))
        pitch = float(options.get("pitch", self.np_random.uniform(-self.initial_tilt_range, self.initial_tilt_range) if random_tilt else 0.))
        if not np.isfinite([roll, pitch]).all() or max(abs(roll), abs(pitch)) > self.initial_tilt_limit:
            raise ValueError("reset roll/pitch must be finite radians within +/-5 degrees")
        self.initial_roll_pitch = np.array([roll, pitch])
        # The parked tool poses use the shared world frame independently of
        # box orientation and must fit the Cartesian command workspace.
        for attempt in range(1024):
            offset = (self._sample_initial_offset() if randomized else
                      np.asarray(options.get("offset", self.fixed_offset), dtype=float))
            yaw = self._sample_initial_yaw() if random_yaw else float(options.get("yaw", self.fixed_yaw))
            if offset.shape != (2,) or not np.all(np.isfinite(offset)) or not np.isfinite(yaw):
                raise ValueError("reset offset must be two finite coordinates; yaw must be finite")
            if abs(yaw) > self.yaw_limit:
                raise ValueError(f"reset yaw must lie within +/-{math.degrees(self.yaw_limit):g} degrees")
            rotation = np.eye(3)
            targets = np.asarray(type(self).initial_tip_positions, dtype=float).copy()
            inside = not (np.any(targets < self.workspace_low - 1e-9) or np.any(targets > self.workspace_high + 1e-9))
            if inside:
                break
            if not (randomized or random_yaw):
                raise ValueError("reset places the parked tools outside their workspace")
        else:
            raise ValueError("Could not sample an offset/yaw inside both arms' existing workspace.")
        self.initial_pose_rejections = attempt
        self.initial_offset = offset.copy()
        self.initial_randomized = randomized
        self.initial_yaw = yaw
        self.initial_yaw_randomized = random_yaw
        box_orientation = box_rotation(roll, pitch, yaw)
        # Place the lowest rotated bottom corner on the lower box's rim plane.
        # Gravity remains active: an unsupported tilt can naturally settle.
        from .model import BOX_SIZE
        bottom = np.array([[x, y, 0.] for x in (-BOX_SIZE[0]/2, BOX_SIZE[0]/2)
                           for y in (-BOX_SIZE[1]/2, BOX_SIZE[1]/2)])
        initial_z = self.control_height - float(np.min((bottom @ box_orientation.T)[:, 2]))
        top = bottom.copy()
        top[:,2] = BOX_SIZE[2]
        box_top = initial_z + float(np.max((top @ box_orientation.T)[:,2]))
        # Extra 22 mm covers the visible finger mesh below its pad-centre site.
        targets[:,2] = max(float(targets[0,2]), box_top+self.initial_finger_clearance_m+.022)
        parking_rotation = self._rotation_increment(np.array([self.initial_tool_tilt_radians,0.,0.]))
        self.initial_tip_rotations = parking_rotation @ self.rest_tip_rotations
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.box_qadr:self.box_qadr+7] = [*offset, initial_z, *quaternion(box_orientation)]
        self.tip_targets[:] = targets
        self.tip_target_rotations[:] = self.initial_tip_rotations
        self.gripper_targets[:] = np.clip(self.gripper_opening,
                                          self.gripper_limits[:, 0], self.gripper_limits[:, 1])
        self.ik_clipped = False
        self.ik_q[:] = [ARM_HOME[arm] for arm in ARM_NAMES]
        for i in range(2):
            self.data.qpos[self.arm_qids[i]] = self.ik_q[i]
            self.data.qpos[self.finger_qids[i]] = [self.gripper_targets[i], -self.gripper_targets[i]]
            self.data.ctrl[self.gripper_aids[i]] = self.gripper_targets[i]
        for _ in range(3):
            if np.any(targets > self.workspace_high) or np.any(targets < self.workspace_low):
                raise ValueError("Safe initial finger height lies outside the Cartesian workspace")
            self.tip_targets[:] = targets
            for i, solver in enumerate(self.ik_solvers):
                q = solver.solve(self.data.qpos, self.ik_q[i], targets[i], max_iterations=250,
                                 rotation=self.initial_tip_rotations[i])
                if not solver.converged:
                    raise RuntimeError(f"{ARM_NAMES[i]} safe initial IK did not converge: {solver.position_error:.4f} m")
                self.ik_q[i] = q
                self.data.qpos[self.arm_qids[i]] = q
                self.data.ctrl[self.arm_aids[i]] = q
            mujoco.mj_forward(self.model, self.data)
            deficit = self.box_highest_z+self.initial_finger_clearance_m-np.min(self.finger_lowest_heights)
            if deficit <= 1e-7:
                break
            targets[:,2] += deficit+.001
        else:
            raise RuntimeError("Could not place both visible finger tips above the box")
        self.initial_tip_positions = targets.copy()
        self.initial_box_top_height = self.box_highest_z
        self.initial_finger_lowest_heights = self.finger_lowest_heights.copy()
        self.steps = self.stable_steps = 0
        self.finish_requested = False
        self.finish_reason = None
        self.failure_reason = None
        self.success = False
        self._set_phase("control")
        self.contact_forces[:] = self.peak_forces[:] = 0
        self.had_contact[:] = False
        self.unintended_contact_count = 0
        self.unintended_contact_peak = 0.
        self.unintended_contact_pairs.clear()
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def _read_contact_forces(self):
        values = np.zeros(2)
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self.robot_geoms or g2 in self.robot_geoms):
                intended = any((g1 in geoms and g2 in geoms)
                               or (g1 in geoms and g2 in self.upper_geoms)
                               or (g2 in geoms and g1 in self.upper_geoms)
                               for geoms in self.probe_geoms)
                if not intended:
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    if wrench[0] > .1:
                        self.unintended_contact_count += 1
                        self.unintended_contact_peak = max(self.unintended_contact_peak, float(wrench[0]))
                        self.unintended_contact_pairs.add((self.model.geom(g1).name, self.model.geom(g2).name))
            if g1 not in self.upper_geoms and g2 not in self.upper_geoms:
                continue
            other = g2 if g1 in self.upper_geoms else g1
            for arm, geoms in enumerate(self.probe_geoms):
                if other in geoms:
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    values[arm] += max(0., wrench[0])
        return values

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain 14 finite values: inner XYZ/rxryrz/opening, outer XYZ/rxryrz/opening")
        if self.phase in ("complete", "failed"):
            info = self._info()
            return self._observation(), self._reward(info), bool(info["success"] or info["failure"]), not (info["success"] or info["failure"]), info
        if self.finish_requested or self.steps >= self.max_steps:
            if not self.finish_requested:
                self.request_finish("time_limit")
            self._advance_phase(self._info())
            info = self._info()
            terminated = bool(info["success"] or info["failure"])
            return self._observation(), self._reward(info), terminated, not terminated, info
        self.ik_clipped = False
        commands = np.clip(action, self.action_space.low, self.action_space.high).reshape(2, 7)
        targets = self.tip_targets + commands[:, :3] * .001
        rotations = np.asarray([
            self._rotation_increment(command[3:6] * self.rotation_action_scale) @ rotation
            for command, rotation in zip(commands, self.tip_target_rotations)])
        self.set_targets(targets, rotations)
        self.gripper_targets[:] = (self.gripper_limits[:, 0]
                                   + commands[:, 6] * np.diff(self.gripper_limits, axis=1).ravel())
        for i in range(2):
            self.data.ctrl[self.arm_aids[i]] = self.ik_q[i]
            self.data.ctrl[self.gripper_aids[i]] = self.gripper_targets[i]
        forces = np.zeros(2)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            current = self._read_contact_forces()
            forces += current / self.frame_skip
            self.peak_forces = np.maximum(self.peak_forces, current)
            self.had_contact |= current > .03
        mujoco.mj_forward(self.model, self.data)
        self.contact_forces[:] = forces
        self.steps += 1
        self.phase_steps += 1
        info = self._info()
        self._advance_phase(info)
        if self.steps >= self.max_steps and self.phase not in ("complete", "failed"):
            self.request_finish("time_limit")
            self._advance_phase(self._info())
        info = self._info()
        terminated = bool(info["success"] or info["failure"])
        truncated = self.phase == "complete" and not terminated
        return self._observation(), self._reward(info), terminated, truncated, info

    def _reward(self, info):
        return (-info["xy_error"] * 10 - abs(info["height_error"]) * 10
                - info["yaw_error"] + (10. if info["success"] else 0.))

    def _info(self):
        pos, yaw = self.pose
        rot = self.data.xmat[self.box_id].reshape(3, 3)
        tilt = math.acos(np.clip(rot[2, 2], -1, 1))
        finite = bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))
        grip_positions = self.data.site_xpos[self.grip_ids]
        outside = (np.any(grip_positions < self.workspace_low - .025)
                   or np.any(grip_positions > self.workspace_high + .025))
        failure = (not finite or tilt > .20 or outside
                   or self.unintended_contact_peak > 5.
                   or self.failure_reason is not None)
        target_height = self.control_height
        lifted_height = float(pos[2] - UPPER_REST_Z)
        return {"xy_error": float(np.linalg.norm(pos[:2])), "yaw_error": abs(yaw),
                "tip_poses": self.tip_poses, "pose_reference_frame": "world",
                "initial_roll_pitch": self.initial_roll_pitch.copy(),
                "initial_box_top_height": self.initial_box_top_height,
                "initial_finger_lowest_heights": self.initial_finger_lowest_heights.copy(),
                "gripper_positions": self.gripper_positions,
                "gripper_targets": self.gripper_targets.copy(),
                "tip_target_rotations": self.tip_target_rotations.copy(),
                "initial_yaw": self.initial_yaw, "initial_yaw_randomized": self.initial_yaw_randomized,
                "initial_yaw_low": self.initial_yaw_low, "initial_yaw_high": self.initial_yaw_high,
                "minimum_initial_yaw": self.minimum_initial_yaw,
                "initial_pose_rejections": self.initial_pose_rejections,
                "yaw_convention": "clockwise_positive_viewed_from_above",
                "initial_offset": self.initial_offset.copy(),
                "initial_randomized": bool(self.initial_randomized),
                "initial_offset_low": np.asarray(self.initial_offset_low, dtype=float),
                "initial_offset_high": np.asarray(self.initial_offset_high, dtype=float),
                "minimum_initial_distance": self.minimum_initial_distance,
                "success": bool(self.success), "failure": bool(failure),
                "phase": self.phase, "phase_steps": self.phase_steps,
                "control_enabled": self.phase == "control", "target_height": target_height,
                "rest_height": UPPER_REST_Z, "lifted_height": lifted_height,
                "height_error": float(pos[2] - target_height),
                "finish_requested": bool(self.finish_requested),
                "pending_finish": bool(self.finish_requested and self.phase not in ("complete", "failed")),
                "manual_finish": bool(self.finish_requested and self.finish_reason not in ("aligned", "time_limit")),
                "finish_reason": self.finish_reason, "failure_reason": self.failure_reason,
                "displacement": pos[:2].copy(), "yaw": yaw, "tilt": tilt,
                "height": float(pos[2]), "contact_forces": self.contact_forces.copy(),
                "peak_forces": self.peak_forces.copy(), "had_contact": self.had_contact.copy(),
                "unintended_contact_count": self.unintended_contact_count,
                "unintended_contact_peak": self.unintended_contact_peak,
                "ik_clipped": bool(self.ik_clipped), "time": float(self.data.time)}

    def _observation(self):
        pos, yaw = self.pose
        return {"box_pose": self.data.qpos[self.box_qadr:self.box_qadr+7].copy(),
                "box_velocity": self.data.qvel[self.box_dadr:self.box_dadr+6].copy(),
                "joint_positions": self.data.qpos[self.qids].copy(),
                "joint_velocities": self.data.qvel[self.dofs].copy(),
                "tip_positions": self.tip_positions, "tip_targets": self.tip_targets.copy(),
                "alignment_error": np.array([*pos[:2], yaw]),
                "contact_forces": self.contact_forces.copy(),
                "gripper_positions": self.gripper_positions,
                "gripper_targets": self.gripper_targets.copy(),
                "tip_quaternions": self.tip_quaternions}

    def render(self, camera="overview", render_size=None):
        if camera not in ("overview", "closeup"):
            raise ValueError("camera must be 'overview' or 'closeup'")
        size = self.render_size if render_size is None else render_size
        if type(size) is not int or size < 16 or size % 2:
            raise ValueError("render_size must be an even integer >= 16.")
        key = (camera, size)
        if key not in self._renderers:
            height = size if camera == "overview" else max(2, 2 * round(size * .3))
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=size)
        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
