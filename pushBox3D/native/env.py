"""Box-centred XY/yaw control with a physical automatic 2 cm lift and landing."""

from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from .model import (ARM_NAMES, ARM_HOME, FINGER_OPENING, GRIP_POINTS,
                    TIP_ROTATIONS, UPPER_REST_Z, build_scene, yaw_rotation)
from .kinematics import ArmIK


class StackedBoxesEnv(gym.Env):
    """3 actions: centre dx/dy (1 mm) and world-Z dyaw (0.5 degrees) per tick.

    The lower box is fixed, upper box has a free joint, and both arms have six
    torque-limited position servos. Numerical IK keeps the pad centres halfway
    up the side walls, with their closing axis across the wall thickness.
    Weld constraints keep both tools on those side-wall midpoints. The box
    moves through those constraints; its pose is written only during reset.
    Each episode lifts the measured box by 2 cm before accepting XY/yaw commands,
    then lowers it before either success or a requested partial-episode end.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    lift_height = .02
    xy_tolerance = .005  # Inclusive radial XY distance, not a per-axis bound.
    height_tolerance = .0002
    stage_timeout_seconds = 12.0
    initial_offset_low = (-.040, -.030)
    initial_offset_high = (.040, .030)
    minimum_initial_distance = .020
    initial_yaw_low = math.radians(-8.)
    initial_yaw_high = math.radians(8.)
    minimum_initial_yaw = math.radians(3.)
    yaw_limit = math.radians(12.)
    yaw_action_scale = math.radians(.5)

    def __init__(self, render_mode=None, offset=(.035, -.02), yaw=0.0,
                 max_episode_seconds=120.0, render_size=512):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.render_size = render_size
        self.fixed_offset = np.asarray(offset, dtype=float).copy()
        self.initial_offset = self.fixed_offset.copy()
        self.initial_randomized = False
        self.fixed_yaw = self.initial_yaw = float(yaw)
        self.initial_yaw_randomized = False
        self.model = mujoco.MjModel.from_xml_string(build_scene(offset, yaw))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = int(max_episode_seconds / self.dt)
        self.action_space = spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "box_pose": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float64),
            "box_velocity": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float64),
            "joint_positions": spaces.Box(-np.inf, np.inf, (12,), dtype=np.float64),
            "joint_velocities": spaces.Box(-np.inf, np.inf, (12,), dtype=np.float64),
            "tip_positions": spaces.Box(-np.inf, np.inf, (2, 3), dtype=np.float64),
            "tip_targets": spaces.Box(-np.inf, np.inf, (2, 3), dtype=np.float64),
            "alignment_error": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
            "contact_forces": spaces.Box(0., np.inf, (2,), dtype=np.float64),
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
        self.robot_geoms = set()
        for gid in range(self.model.ngeom):
            body_name = self.model.body(int(self.model.geom_bodyid[gid])).name
            if body_name.startswith(tuple(f"{arm}_" for arm in ARM_NAMES)):
                self.robot_geoms.add(gid)
        self.tip_ids = [self.model.site(f"{arm}_tip").id for arm in ARM_NAMES]
        self.grip_ids = [self.model.site(f"{arm}_grip").id for arm in ARM_NAMES]
        self.attach_ids = [self.model.equality(f"{arm}_attach").id for arm in ARM_NAMES]
        self.grip_offsets = np.asarray(GRIP_POINTS, dtype=float)
        self.rest_tip_rotations = np.asarray([TIP_ROTATIONS[arm] for arm in ARM_NAMES])
        # Preserve the original mounted ARX workspace: box XY ±55/40 mm and
        # 0..35 mm commanded lift. Height feedback uses measured free-body Z
        # to compensate actuator compliance while raising the box by 20 mm.
        grip_low = self.grip_offsets.min(axis=0)
        grip_high = self.grip_offsets.max(axis=0)
        self.workspace_low = grip_low + [-.055, -.040, UPPER_REST_Z]
        self.workspace_high = grip_high + [.055, .040, UPPER_REST_Z + .035]
        self.ik_solvers = [ArmIK(self.model, self.tip_ids[i], self.arm_qids[i],
                                 self.arm_dofs[i], self.arm_jids[i],
                                 np.asarray(TIP_ROTATIONS[ARM_NAMES[i]]))
                           for i in range(2)]
        self.ik_q = np.array([ARM_HOME[arm] for arm in ARM_NAMES])
        self.tip_targets = np.zeros((2, 3))
        self.box_target_center = np.zeros(3)
        self.box_target_yaw = 0.
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

    def set_targets(self, targets):
        """Accept only a rigid pair, interpreted as one box-centred pose."""
        targets = np.asarray(targets, dtype=float)
        if targets.shape != (2, 3) or not np.all(np.isfinite(targets)):
            raise ValueError("targets must be a finite 2x3 array in world metres")
        pair = targets[1] - targets[0]
        yaw = -math.atan2(pair[1], pair[0])
        rotation = np.asarray(yaw_rotation(yaw))
        offsets = self.grip_offsets @ rotation.T
        if not np.allclose(pair, offsets[1] - offsets[0], atol=1e-9, rtol=0):
            raise ValueError("attached grips must retain their rigid spacing and equal height")
        self.set_pose_target(targets[0] - offsets[0], yaw)

    def set_pose_target(self, center, yaw):
        """Rotate both grips and tool frames about a shared box centre.

        A shortened rigid-pose increment handles workspace/IK limits. In
        particular, pure yaw never moves the commanded centre sideways.
        Only joint actuator targets change; this never writes live box qpos.
        """
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
            raise ValueError("centre must be finite world XYZ and yaw must be finite radians")
        bounded_center = center.copy()
        bounded_center[2] = np.clip(center[2], UPPER_REST_Z, self.workspace_high[2] - self.grip_offsets[0, 2])
        bounded_yaw = float(np.clip(yaw, -self.yaw_limit, self.yaw_limit))
        self.ik_clipped |= not np.allclose(center, bounded_center, atol=1e-12, rtol=0) or abs(yaw - bounded_yaw) > 1e-12
        delta = bounded_center - self.box_target_center
        dyaw = bounded_yaw - self.box_target_yaw
        for fraction in (1., .5, .25, .125, .0625):
            candidate_center = self.box_target_center + fraction * delta
            candidate_yaw = self.box_target_yaw + fraction * dyaw
            rotation = np.asarray(yaw_rotation(candidate_yaw))
            candidate = candidate_center + self.grip_offsets @ rotation.T
            if np.any(candidate < self.workspace_low - 1e-12) or np.any(candidate > self.workspace_high + 1e-12):
                continue
            solutions = []
            for i, solver in enumerate(self.ik_solvers):
                solutions.append(solver.solve(self.data.qpos, self.ik_q[i], candidate[i],
                                              rotation=rotation @ self.rest_tip_rotations[i]))
                if not solver.converged:
                    break
            else:
                self.box_target_center[:] = candidate_center
                self.box_target_yaw = candidate_yaw
                self.tip_targets[:] = candidate
                self.ik_q[:] = solutions
                self.ik_clipped |= fraction < 1.
                return
        self.ik_clipped = True

    def _set_phase(self, phase):
        self.phase = phase
        self.phase_steps = 0
        self.stable_steps = 0

    def request_finish(self, reason="manual"):
        """Queue a physical landing without resetting, teleporting or saving.

        Continue calling step([0, 0, 0]) until terminated/truncated. A request
        during the opening lift completes that lift first. A settled landing
        away from the XY goal truncates the episode rather than reporting
        success; an aligned settled landing can still succeed.
        """
        if self.phase in ("complete", "failed"):
            return self._info()
        if not self.finish_requested:
            self.finish_requested = True
            self.finish_reason = str(reason)
        if self.phase == "control":
            self._set_phase("lowering")
        return self._info()

    def _height_command(self):
        """Rate-limited feedback, so 2 cm refers to the actual box height."""
        desired = UPPER_REST_Z + (self.lift_height if self.phase in ("lifting", "control") else 0.)
        measured = self.data.xpos[self.box_id, 2]
        velocity = self.data.qvel[self.box_dadr + 2]
        # Integrating measured error removes the several-mm payload sag. The
        # velocity term damps the feedback; all commanded motion remains at
        # most 1 mm per 20 ms control tick, including landing.
        return float(np.clip(.10 * (desired - measured) - .02 * velocity, -.001, .001))

    def _advance_phase(self, info):
        if info["failure"]:
            self._set_phase("failed")
            return
        if self.phase not in ("control", "complete", "failed") and self.phase_steps * self.dt >= self.stage_timeout_seconds:
            self.failure_reason = f"{self.phase}_timeout"
            self._set_phase("failed")
            return
        velocity = self.data.qvel[self.box_dadr:self.box_dadr + 6]
        slow = np.linalg.norm(velocity) < .015
        aligned = (info["xy_error"] <= self.xy_tolerance and info["yaw_error"] < .01
                   and info["tilt"] < .02 and info["attached"])
        if self.phase == "lifting":
            ready = (abs(info["lifted_height"] - self.lift_height) < self.height_tolerance
                     and abs(velocity[2]) < .003)
            self.stable_steps = self.stable_steps + 1 if ready else 0
            if self.stable_steps >= round(.15 / self.dt):
                self._set_phase("lowering" if self.finish_requested else "control")
        elif self.phase == "control":
            ready = (aligned and slow and abs(info["lifted_height"] - self.lift_height) < .001)
            self.stable_steps = self.stable_steps + 1 if ready else 0
            if self.stable_steps >= round(.2 / self.dt):
                self.request_finish("aligned")
        elif self.phase == "lowering":
            if abs(info["height"] - UPPER_REST_Z) < self.height_tolerance and abs(velocity[2]) < .003:
                self._set_phase("settling")
        elif self.phase == "settling":
            landed = abs(info["height"] - UPPER_REST_Z) < self.height_tolerance
            self.stable_steps = self.stable_steps + 1 if landed and slow else 0
            if self.stable_steps >= round(.5 / self.dt):
                self.success = bool(aligned)
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
        """Seeded random XY/yaw; explicit offset and yaw override each component.

        Reusing a seed reproduces its initial pose. ``randomize=False`` uses
        the constructor's fixed offset, preserving controlled-test entry points.
        """
        super().reset(seed=seed)
        options = options or {}
        randomized = "offset" not in options and bool(options.get("randomize", True))
        random_yaw = bool(options.get("randomize", True)) and "yaw" not in options
        # Larger XY and yaw proposals are coupled by the existing two-arm
        # workspace. Reject a pose rather than clipping its sampled geometry.
        for attempt in range(1024):
            offset = (self._sample_initial_offset() if randomized else
                      np.asarray(options.get("offset", self.fixed_offset), dtype=float))
            yaw = self._sample_initial_yaw() if random_yaw else float(options.get("yaw", self.fixed_yaw))
            if offset.shape != (2,) or not np.all(np.isfinite(offset)) or not np.isfinite(yaw):
                raise ValueError("reset offset must be two finite coordinates; yaw must be finite")
            if abs(yaw) > self.yaw_limit:
                raise ValueError(f"reset yaw must lie within +/-{math.degrees(self.yaw_limit):g} degrees")
            rotation = np.asarray(yaw_rotation(yaw))
            center = np.array([*offset, UPPER_REST_Z])
            targets = center + self.grip_offsets @ rotation.T
            inside = not (np.any(targets < self.workspace_low - 1e-9) or np.any(targets > self.workspace_high + 1e-9))
            if inside:
                break
            if not (randomized or random_yaw):
                raise ValueError("reset offset/yaw places the attached tools outside their shared workspace")
        else:
            raise ValueError("Could not sample an offset/yaw inside both arms' existing workspace.")
        self.initial_pose_rejections = attempt
        self.initial_offset = offset.copy()
        self.initial_randomized = randomized
        self.initial_yaw = yaw
        self.initial_yaw_randomized = random_yaw
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.box_qadr:self.box_qadr+7] = [*offset, UPPER_REST_Z, math.cos(yaw/2), 0, 0, -math.sin(yaw/2)]
        self.tip_targets[:] = targets
        self.box_target_center[:] = center
        self.box_target_yaw = yaw
        self.data.eq_active[self.attach_ids] = True
        self.ik_clipped = False
        self.ik_q[:] = [ARM_HOME[arm] for arm in ARM_NAMES]
        for i in range(2):
            self.data.qpos[self.arm_qids[i]] = self.ik_q[i]
            self.data.qpos[self.finger_qids[i]] = [FINGER_OPENING, -FINGER_OPENING]
            # Start without interpenetration, then command the original
            # parallel fingers closed; the 8 mm wall supplies their stop.
            self.data.ctrl[self.gripper_aids[i]] = 0.
        for i in range(2):
            solver = self.ik_solvers[i]
            q = solver.solve(self.data.qpos, self.ik_q[i], self.tip_targets[i], max_iterations=250,
                             rotation=rotation @ self.rest_tip_rotations[i])
            if not solver.converged:
                raise RuntimeError(f"{ARM_NAMES[i]} ARX L5 home IK did not converge: {solver.position_error:.4f} m")
            self.ik_q[i] = q
            self.data.qpos[self.arm_qids[i]] = q
            self.data.ctrl[self.arm_aids[i]] = q
        self.steps = self.stable_steps = 0
        self.finish_requested = False
        self.finish_reason = None
        self.failure_reason = None
        self.success = False
        self._set_phase("lifting")
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
                intended = any((g1 in geoms and g2 in self.upper_geoms)
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
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain three finite values: dx, dy, dyaw")
        if self.phase in ("complete", "failed"):
            info = self._info()
            return self._observation(), self._reward(info), bool(info["success"] or info["failure"]), not (info["success"] or info["failure"]), info
        self.ik_clipped = False
        delta = np.zeros(3)
        dyaw = 0.
        if self.phase == "control":
            delta[:2] = np.clip(action[:2], -1, 1) * .001
            dyaw = float(np.clip(action[2], -1, 1)) * self.yaw_action_scale
        delta[2] = self._height_command()
        self.set_pose_target(self.box_target_center + delta, self.box_target_yaw + dyaw)
        for i in range(2):
            self.data.ctrl[self.arm_aids[i]] = self.ik_q[i]
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
        attachment_error = float(np.max(np.linalg.norm(
            self.tip_positions - self.data.site_xpos[self.grip_ids], axis=1)))
        attached = bool(np.all(self.data.eq_active[self.attach_ids]))
        grip_positions = self.data.site_xpos[self.grip_ids]
        outside = (np.any(grip_positions < self.workspace_low - .025)
                   or np.any(grip_positions > self.workspace_high + .025))
        failure = (not finite or tilt > .20 or outside or not attached
                   or attachment_error > .01 or self.unintended_contact_peak > 5.
                   or self.failure_reason is not None)
        target_height = UPPER_REST_Z + (self.lift_height if self.phase in ("lifting", "control") else 0.)
        lifted_height = float(pos[2] - UPPER_REST_Z)
        return {"xy_error": float(np.linalg.norm(pos[:2])), "yaw_error": abs(yaw),
                "box_target_center": self.box_target_center.copy(), "box_target_yaw": self.box_target_yaw,
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
                # Lift is actual rise from the original resting box height.
                # Errors are signed: positive means above the named target.
                "rest_height": UPPER_REST_Z, "lifted_height": lifted_height,
                "lift_error": lifted_height - self.lift_height,
                "landing_height_error": lifted_height,
                # The active Z target is rest+20 mm while lifting/control,
                # then the original resting height during lowering/settling.
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
                "attached": attached, "attachment_error": attachment_error,
                "ik_clipped": bool(self.ik_clipped), "time": float(self.data.time)}

    def _observation(self):
        pos, yaw = self.pose
        return {"box_pose": self.data.qpos[self.box_qadr:self.box_qadr+7].copy(),
                "box_velocity": self.data.qvel[self.box_dadr:self.box_dadr+6].copy(),
                "joint_positions": self.data.qpos[self.qids].copy(),
                "joint_velocities": self.data.qvel[self.dofs].copy(),
                "tip_positions": self.tip_positions, "tip_targets": self.tip_targets.copy(),
                "alignment_error": np.array([*pos[:2], yaw]),
                "contact_forces": self.contact_forces.copy()}

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
