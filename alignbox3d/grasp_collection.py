"""Contact-only grasp/carry/release expert; all motion is a 14-D action."""
from dataclasses import asdict, dataclass
import math
import numpy as np

from common.recording import EpisodeRecorder
from .geometry import quaternion, rotation_vector, smoothstep
from .native.model import BOX_SIZE, GRIP_POINTS
from .stage_labels import EXPERT_STAGES, EXPERT_STAGE_LABEL_IDS, STAGE_LABELS, sample_stage_instructions
FREE_TOOL_STAGES = frozenset(("approach_above", "approach_rim", "orient_grasp",
                              "withdraw_orient", "withdraw_above", "park", "recover_contact"))
# Require measured stillness for 10 policy intervals (1/3 s at the default 30 Hz).
# Per-frame thresholds are speed * dt: 50 um and 0.0167 deg at 30 Hz.
TOOL_STILL_FRAMES = 10
TOOL_STILL_LINEAR_SPEED_M_S = .0015
TOOL_STILL_ANGULAR_SPEED_RAD_S = math.radians(.5)
TOOL_SETTLE_TIMEOUT_S = 2.5


@dataclass(frozen=True)
class GraspConfig:
    point_y_range_m: float = .010
    point_z_range_m: float = .005
    pair_y_difference_m: float = .002
    pair_z_difference_m: float = .005
    angle_jitter_degrees: float = 1.5
    independent_angle_degrees: float = .25
    lift_m: float = .020
    approach_speed_m_s: float = .050
    withdraw_speed_m_s: float = .050
    free_angular_speed_degrees_s: float = 24.
    contact_speed_m_s: float = .015
    lift_speed_m_s: float = .020
    carry_speed_m_s: float = .020
    lower_speed_m_s: float = .020
    angular_speed_degrees_s: float = 12.
    limit_motion_acceleration: bool = False
    linear_acceleration_m_s2: float = .030
    angular_acceleration_degrees_s2: float = 20.
    gripper_speed_m_s: float = .004
    gripper_min_ramp_seconds: float = 2.5
    settle_seconds: float = .4
    stage_timeout_s: float = 40.

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("Grasp configuration values must be finite and nonnegative")
        if min(self.approach_speed_m_s, self.withdraw_speed_m_s, self.free_angular_speed_degrees_s,
               self.contact_speed_m_s, self.lift_speed_m_s,
               self.carry_speed_m_s, self.lower_speed_m_s, self.angular_speed_degrees_s,
               self.linear_acceleration_m_s2, self.angular_acceleration_degrees_s2,
               self.gripper_speed_m_s, self.gripper_min_ramp_seconds,
               self.settle_seconds, self.stage_timeout_s) <= 0:
            raise ValueError("Speeds, acceleration limits, durations and timeout must be positive")
        if self.pair_y_difference_m > 2*self.point_y_range_m or self.pair_z_difference_m > 2*self.point_z_range_m:
            raise ValueError("Pair differences must fit inside the point ranges")
        if self.point_y_range_m >= BOX_SIZE[1]/2-.02 or self.point_z_range_m >= BOX_SIZE[2]/2-.02:
            raise ValueError("Sampled contacts must stay on the side-wall interior, clear of its edges")
        if max(self.approach_speed_m_s, self.withdraw_speed_m_s, self.contact_speed_m_s,
               self.lift_speed_m_s, self.carry_speed_m_s, self.lower_speed_m_s) > .050:
            raise ValueError("Cartesian speed must not exceed 50 mm/s")
        if max(self.angular_speed_degrees_s, self.free_angular_speed_degrees_s) > 25.:
            raise ValueError("Angular speed must not exceed 25 degrees/s")


def sample_grasp(seed, config=None):
    """Correlated samples in the box frame: bound BOTH points and their mismatch."""
    config = config or GraspConfig()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 71]))
    points = np.asarray(GRIP_POINTS).copy()
    for axis, spread, difference in ((1, config.point_y_range_m, config.pair_y_difference_m),
                                     (2, config.point_z_range_m, config.pair_z_difference_m)):
        mismatch = rng.uniform(-difference, difference)
        centre = rng.uniform(-spread + abs(mismatch)/2, spread - abs(mismatch)/2)
        points[:, axis] += centre + np.array([-mismatch/2, mismatch/2])
    shared = rng.uniform(-config.angle_jitter_degrees, config.angle_jitter_degrees, 3)
    independent = rng.uniform(-config.independent_angle_degrees, config.independent_angle_degrees, (2, 3))
    return {"points_box_m": points, "angle_vectors_box_rad": np.deg2rad(shared + independent)}


class GraspFailure(RuntimeError):
    pass


class GraspCarryExpert:
    def __init__(self, environment, seed, *, config=None, step=None, max_steps=None,
                 recovery=False, recovery_offset_m=.010, recovery_stage="approach",
                 contact_error_height_m=.020):
        self.environment = environment
        self.native = environment.environment
        self.config = config or GraspConfig()
        self.seed = int(seed)
        self.stage_instructions = sample_stage_instructions(self.seed)
        self.record_step = environment.step if step is None else step
        if recovery_stage not in ("approach", "carry", "contact_error"):
            raise ValueError("recovery_stage must be approach, carry or contact_error")
        if not math.isfinite(contact_error_height_m) or not .010 <= contact_error_height_m <= .025:
            raise ValueError("contact_error_height_m must be within 10-25 mm")
        self.recovery = bool(recovery and (recovery_stage == "contact_error" or recovery_offset_m > 0))
        self.recovery_stage = recovery_stage if self.recovery else None
        self.recovery_offset_m = float(recovery_offset_m)
        self.contact_error_height_m = float(contact_error_height_m)
        self.recovery_start = None
        self.step_callback = environment.step if self.recovery else self.record_step
        self.max_steps = max_steps if max_steps is not None else math.ceil(100 / self.native.dt)
        self.frames = 0
        self.stage_counts = {}
        self.stage = "initial_settle"
        self.trace = []
        self.peak_lift_m = 0.
        self.previous_linear_velocity = np.zeros((2, 3))
        self.previous_angular_velocity = np.zeros((2, 3))
        self.stage_motion = {}
        self.sample = sample_grasp(seed, self.config)
        self.initial_box_pose = self.native.data.qpos[self.native.box_qadr:self.native.box_qadr+7].copy().tolist()
        if hasattr(environment, "collection_metadata"):
            environment.collection_metadata.update(
                recovery_mode_id={None: 0, "approach": 1, "carry": 2,
                                  "contact_error": 3}[self.recovery_stage],
                grasp_points_box_m=self.sample["points_box_m"].tolist(),
                grasp_angle_vectors_box_rad=self.sample["angle_vectors_box_rad"].tolist())
        self.grasp_relative_positions = self.grasp_relative_rotations = None

    def box_pose(self):
        n = self.native
        rotation = n.data.xmat[n.box_id].reshape(3, 3).copy()
        centre = n.pose[0] + rotation @ np.array([0., 0., BOX_SIZE[2]/2])
        return centre, rotation

    def stage_start(self, name):
        self.stage = name
        self.stage_start_z = float(self.native.pose[0][2])
        label_id = EXPERT_STAGE_LABEL_IDS[EXPERT_STAGES.index(name)]
        self.environment.language_stage_id = label_id
        self.environment.instruction = self.stage_instructions[label_id]
        if hasattr(self.environment, "collection_metadata"):
            self.environment.collection_metadata["expert_stage_id"] = EXPERT_STAGES.index(name)
        centre, rotation = self.box_pose()
        self.trace.append({"stage": name, "language_stage_id": label_id,
                           "language_label": STAGE_LABELS[label_id], "frame": self.frames,
                           "instruction": self.stage_instructions[label_id],
                           "box_center_world_m": centre.tolist(),
                           "box_quaternion_wxyz": quaternion(rotation).tolist()})

    def start_recovery_recording(self):
        """Begin a new, continuous expert suffix at the disturbed state."""
        self.frames = 0
        self.stage_counts = {}
        self.trace = []
        self.stage_motion = {}
        self.step_callback = self.record_step

    @staticmethod
    def limit_norm(vectors, maximum):
        vectors = np.asarray(vectors, dtype=float)
        return vectors * np.minimum(1., maximum/np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12))

    def speed_limit(self):
        return {"descend_to_grasp": self.config.contact_speed_m_s,
                "close_grippers": self.config.contact_speed_m_s,
                "lift": self.config.lift_speed_m_s,
                "carry_align": self.config.carry_speed_m_s,
                "lower": self.config.lower_speed_m_s,
                "release": self.config.lower_speed_m_s,
                "withdraw_rim": self.config.contact_speed_m_s,
                "withdraw_orient": self.config.withdraw_speed_m_s,
                "withdraw_above": self.config.withdraw_speed_m_s,
                "park": self.config.withdraw_speed_m_s}.get(self.stage, self.config.approach_speed_m_s)

    def angular_speed_limit(self):
        return (self.config.free_angular_speed_degrees_s if self.stage in FREE_TOOL_STAGES
                else self.config.angular_speed_degrees_s)

    def tick(self, positions, rotations, opening, *, feedback=3., allow_complete=False):
        """Track poses with a feedback gain in 1/s, independent of sample rate."""
        n, c = self.native, self.config
        if self.frames >= self.max_steps:
            raise GraspFailure("episode_step_limit")
        action = n.hold_action().reshape(2, 7)
        desired_velocity = self.limit_norm(feedback*(positions-n.tip_positions), self.speed_limit())
        linear_velocity = (self.previous_linear_velocity + self.limit_norm(
            desired_velocity-self.previous_linear_velocity, c.linear_acceleration_m_s2*n.dt)
            if c.limit_motion_acceleration else desired_velocity)
        action[:, :3] = linear_velocity*n.dt/n.translation_action_scale
        for arm in range(2):
            desired_angular_velocity = self.limit_norm(
                feedback*rotation_vector(rotations[arm]@n.tip_rotations[arm].T),
                math.radians(self.angular_speed_limit()))
            angular_velocity = (self.previous_angular_velocity[arm] + self.limit_norm(
                desired_angular_velocity-self.previous_angular_velocity[arm],
                math.radians(c.angular_acceleration_degrees_s2)*n.dt)
                if c.limit_motion_acceleration else desired_angular_velocity)
            action[arm, 3:6] = angular_velocity*n.dt/n.rotation_action_scale
            self.previous_angular_velocity[arm] = angular_velocity
        self.previous_linear_velocity[:] = linear_velocity
        ranges = np.diff(n.gripper_limits, axis=1).ravel()
        target = n.gripper_limits[:, 0] + np.clip(opening, 0., 1.)*ranges
        next_target = n.gripper_targets + np.clip(target-n.gripper_targets,
                                                  -c.gripper_speed_m_s*n.dt, c.gripper_speed_m_s*n.dt)
        action[:, 6] = (next_target-n.gripper_limits[:, 0])/ranges
        transition = self.step_callback(self.environment.native_action_to_interface(action.ravel()))
        self.frames += 1
        self.stage_counts[self.stage] = self.stage_counts.get(self.stage, 0) + 1
        self.peak_lift_m = max(self.peak_lift_m, float(n.pose[0][2] - n.control_height))
        velocity = n.data.qvel[n.box_dadr:n.box_dadr+3]
        metrics = self.stage_motion.setdefault(self.stage, {"peak_box_speed_m_s": 0., "peak_abs_box_vz_m_s": 0., "max_box_rise_m": 0.})
        metrics["peak_box_speed_m_s"] = max(metrics["peak_box_speed_m_s"], float(np.linalg.norm(velocity)))
        metrics["peak_abs_box_vz_m_s"] = max(metrics["peak_abs_box_vz_m_s"], abs(float(velocity[2])))
        metrics["max_box_rise_m"] = max(metrics["max_box_rise_m"], float(n.pose[0][2])-self.stage_start_z)
        if transition.info["failure"]:
            raise GraspFailure(f"physics_failure:{self.stage}:{transition.info.get('failure_reason')}:{sorted(n.unintended_contact_pairs)}")
        if transition.done and not allow_complete:
            raise GraspFailure(f"premature_completion:{self.stage}")
        return transition

    def move_tools(self, name, positions, rotations, opening, *, speed=None):
        n, c = self.native, self.config
        self.stage_start(name)
        start_positions, start_rotations = n.tip_positions.copy(), n.tip_rotations.copy()
        distance = float(np.max(np.linalg.norm(positions - start_positions, axis=1)))
        angle = max(np.linalg.norm(rotation_vector(end @ start.T))
                    for end, start in zip(rotations, start_rotations))
        duration = max(.3, 1.875*distance/(speed or self.speed_limit()),
                       1.875*angle/math.radians(self.angular_speed_limit()))
        feedback = 9. if name in FREE_TOOL_STAGES else 3.
        if duration > c.stage_timeout_s:
            raise GraspFailure(f"planned_stage_timeout:{name}")
        for index in range(max(1, math.ceil(duration/n.dt))):
            fraction = smoothstep((index + 1)*n.dt/duration)
            target_positions = start_positions + fraction*(positions - start_positions)
            target_rotations = np.asarray([n._interpolate_rotation(start, end, fraction)
                                           for start, end in zip(start_rotations, rotations)])
            self.tick(target_positions, target_rotations, opening, feedback=feedback)
        still_frames = 0
        for _ in range(max(TOOL_STILL_FRAMES, math.ceil(TOOL_SETTLE_TIMEOUT_S / n.dt))):
            previous_positions = n.tip_positions.copy()
            previous_rotations = n.tip_rotations.copy()
            self.tick(positions, rotations, opening, feedback=feedback)
            actual_positions, actual_rotations = n.tip_positions, n.tip_rotations
            reached = (np.max(np.linalg.norm(actual_positions - positions, axis=1)) < .001
                    and max(np.linalg.norm(rotation_vector(end @ actual.T))
                            for end, actual in zip(rotations, actual_rotations)) < math.radians(.5))
            still = (np.max(np.linalg.norm(actual_positions - previous_positions, axis=1))
                     < TOOL_STILL_LINEAR_SPEED_M_S * n.dt
                     and max(np.linalg.norm(rotation_vector(actual @ previous.T))
                             for actual, previous in zip(actual_rotations, previous_rotations))
                     < TOOL_STILL_ANGULAR_SPEED_RAD_S * n.dt)
            still_frames = still_frames + 1 if reached and still else 0
            if still_frames >= TOOL_STILL_FRAMES:
                return
        raise GraspFailure(f"tool_tracking_timeout:{name}")

    def follow_carry_path(self, name, centre, rotation):
        n, c = self.native, self.config
        self.stage_start(name)
        start, start_rotation = self.box_pose()
        duration = max(.5, 1.875*np.linalg.norm(centre-start)/self.speed_limit(),
                       1.875*np.linalg.norm(rotation_vector(rotation@start_rotation.T))
                       /math.radians(c.angular_speed_degrees_s))
        if duration > c.stage_timeout_s:
            raise GraspFailure(f"planned_stage_timeout:{name}")
        stable = 0
        for index in range(math.ceil((duration + 5.)/n.dt)):
            fraction = smoothstep(min(1., (index + 1)*n.dt/duration))
            target_centre = start + fraction*(centre-start)
            target_rotation = n._interpolate_rotation(start_rotation, rotation, fraction)
            actual_centre, actual_rotation = self.box_pose()
            # Apply one measured rigid-body error to both tools. Re-estimating
            # the contact transform avoids stacking a second integral loop on
            # top of the already incremental position commands.
            relative_positions = (n.tip_positions-actual_centre) @ actual_rotation
            relative_rotations = actual_rotation.T @ n.tip_rotations
            positions = target_centre + relative_positions @ target_rotation.T
            rotations = target_rotation @ relative_rotations
            self.tick(positions, rotations, 0.)
            actual_centre, actual_rotation = self.box_pose()
            velocity = n.data.qvel[n.box_dadr:n.box_dadr+6]
            ready = (fraction == 1 and np.linalg.norm(actual_centre-centre) < .0015
                    and np.linalg.norm(rotation_vector(rotation@actual_rotation.T)) < math.radians(.6)
                    and np.linalg.norm(velocity[:3]) < .003 and np.linalg.norm(velocity[3:]) < .02
                    and np.max(np.linalg.norm(self.previous_linear_velocity, axis=1)) < .004)
            stable = stable+1 if ready else 0
            if stable >= math.ceil(c.settle_seconds/n.dt):
                return
        raise GraspFailure(f"box_tracking_timeout:{name}")

    def move_grippers(self, name, positions, rotations, opening):
        n, c = self.native, self.config
        self.stage_start(name)
        ranges = np.diff(n.gripper_limits, axis=1).ravel()
        start = n.hold_action().reshape(2, 7)[:, 6].astype(float)
        target = np.broadcast_to(np.asarray(opening, dtype=float), (2,))
        duration = max(c.gripper_min_ramp_seconds,
                       1.875*float(np.max(np.abs(target-start)*ranges))/c.gripper_speed_m_s)
        for index in range(math.ceil(duration/n.dt)):
            fraction = smoothstep((index+1)*n.dt/duration)
            self.tick(positions, rotations, start+fraction*(target-start))
        stable = 0
        for _ in range(math.ceil(2.5/n.dt)):
            self.tick(positions, rotations, target)
            velocity = n.data.qvel[n.box_dadr:n.box_dadr+6]
            reached = np.max(np.abs(n.hold_action().reshape(2, 7)[:, 6]-target)) < 1e-5
            stable = stable+1 if reached and np.linalg.norm(velocity) < .003 else 0
            if stable >= math.ceil(c.settle_seconds/n.dt):
                if self.stage_motion[name]["max_box_rise_m"] > .0015:
                    raise GraspFailure(f"unexpected_box_rise:{name}")
                if name == "close_grippers" and self.stage_motion[name]["peak_box_speed_m_s"] > .025:
                    raise GraspFailure("grasp_impulse")
                return
        raise GraspFailure(f"gripper_settle_timeout:{name}")

    def grasp_targets(self):
        """Replan from the measured box pose, including after a bad contact."""
        n = self.native
        body = n.pose[0].copy()
        rotation = n.data.xmat[n.box_id].reshape(3, 3).copy()
        points = self.sample["points_box_m"]
        positions = body + points @ rotation.T
        local_rotations = np.asarray([n._rotation_increment(vector) @ rest
                                      for vector, rest in zip(self.sample["angle_vectors_box_rad"], n.rest_tip_rotations)])
        grasp_rotations = rotation @ local_rotations
        approach_rotations = rotation @ n._rotation_increment(np.array([math.radians(20.), 0., 0.])) @ local_rotations
        # Test full tool poses without touching the live physics state.
        for candidate_positions, candidate_rotations in (
                (positions+[0,0,.09], approach_rotations),
                (positions+[0,0,.03], grasp_rotations), (positions, grasp_rotations)):
            for arm, solver in enumerate(n.ik_solvers):
                solver.solve(n.data.qpos, n.ik_q[arm], candidate_positions[arm],
                             max_iterations=250, rotation=candidate_rotations[arm])
                if not solver.converged:
                    raise GraspFailure("sampled_grasp_not_reachable")
        return positions, grasp_rotations, approach_rotations

    def create_contact_error(self, positions, grasp_rotations, approach_rotations, opening):
        """Press one misplaced, partly closed gripper against the upper box."""
        n = self.native
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 78]))
        arm = int(rng.integers(2))
        height_error = float(rng.uniform(max(.010, self.contact_error_height_m*.75),
                                         self.contact_error_height_m))
        self.move_tools("approach_above", positions+[0, 0, .09], approach_rotations, opening)
        self.move_tools("approach_rim", positions+[0, 0, .03], approach_rotations, opening)
        self.move_tools("orient_grasp", positions+[0, 0, .03], grasp_rotations, opening)
        wrong = positions.copy()
        wrong[arm, 2] += height_error
        self.move_tools("descend_to_grasp", wrong, grasp_rotations, opening,
                        speed=self.config.contact_speed_m_s)
        openings = np.full(2, opening)
        openings[arm] = 0.
        self.move_grippers("close_grippers", wrong, grasp_rotations, openings)
        forces = n.contact_forces.copy()
        if forces[arm] < 1. or forces[1-arm] > .3:
            raise GraspFailure("contact_error_not_single_sided")
        self.recovery_start = {"stage": "contact_error", "arm": arm,
                               "height_error_m": height_error,
                               "contact_forces_N": forces.tolist(),
                               "wrong_tool_positions_world_m": n.tip_positions.tolist(),
                               "box_pose_world": n.data.qpos[n.box_qadr:n.box_qadr+7].copy().tolist()}

    def recover_contact(self, opening):
        """Open, clear the rim, reorient, and park before replanning."""
        n = self.native
        self.move_grippers("recover_contact", n.tip_positions.copy(),
                           n.tip_rotations.copy(), opening)
        rim_positions = n.tip_positions.copy()
        rim_positions[:, 2] += .010
        self.move_tools("recover_contact", rim_positions, n.tip_rotations.copy(),
                        opening, speed=self.config.contact_speed_m_s)
        withdrawal_rotations = (n._rotation_increment(np.array([math.radians(20.), 0., 0.]))
                                @ n.tip_rotations.copy())
        self.move_tools("recover_contact", rim_positions, withdrawal_rotations, opening)
        above_positions = rim_positions.copy()
        above_positions[:, 2] = n.control_height+.15
        self.move_tools("recover_contact", above_positions, withdrawal_rotations, opening)
        self.move_tools("recover_contact", np.asarray(n.initial_tip_positions),
                        n.initial_tip_rotations, opening)

    def run(self):
        n = self.native
        n.terminate_on_alignment = False
        self.stage_start("initial_settle")
        stable = 0
        for index in range(math.ceil(2.5/n.dt)):
            self.tick(np.asarray(n.initial_tip_positions), n.initial_tip_rotations,
                      n.hold_action().reshape(2, 7)[:, 6])
            velocity = n.data.qvel[n.box_dadr:n.box_dadr+6]
            stable = stable+1 if np.linalg.norm(velocity) < .003 else 0
            if (index + 1)*n.dt >= 1/3 and stable >= math.ceil((1/6)/n.dt):
                break
        else:
            raise GraspFailure("initial_box_not_settled")
        opening = .012/.044
        positions, grasp_rotations, approach_rotations = self.grasp_targets()
        if self.recovery_stage == "contact_error":
            self.create_contact_error(positions, grasp_rotations, approach_rotations, opening)
            self.start_recovery_recording()
            self.recover_contact(opening)
            positions, grasp_rotations, approach_rotations = self.grasp_targets()
        above_positions = positions+[0,0,.09]
        if self.recovery_stage == "approach":
            # Construct a plausible off-path state before recording. Including
            # the mistake-making actions would teach behavioural cloning to
            # make that mistake, so the saved episode starts at correction.
            self.step_callback = self.environment.step
            self.move_tools("approach_above", above_positions, approach_rotations, opening)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 76]))
            arm = int(rng.integers(2))
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            magnitude = rng.uniform(self.recovery_offset_m*.5, self.recovery_offset_m)
            disturbed = above_positions.copy()
            disturbed[arm] += direction*magnitude
            disturbed[arm] = np.clip(disturbed[arm], n.workspace_low+.002, n.workspace_high-.002)
            self.move_tools("approach_above", disturbed, approach_rotations, opening)
            self.recovery_start = {"stage": "approach", "arm": arm,
                                   "target_tool_positions_world_m": above_positions.tolist(),
                                   "actual_tool_positions_world_m": n.tip_positions.tolist()}
            self.start_recovery_recording()
            self.move_tools("approach_above", above_positions, approach_rotations, opening)
        else:
            self.move_tools("approach_above", above_positions, approach_rotations, opening)
        self.move_tools("approach_rim", positions+[0,0,.03], approach_rotations, opening)
        self.move_tools("orient_grasp", positions+[0,0,.03], grasp_rotations, opening)
        self.move_tools("descend_to_grasp", positions, grasp_rotations, opening, speed=self.config.contact_speed_m_s)
        self.move_grippers("close_grippers", positions, grasp_rotations, 0.)
        if np.min(n.contact_forces) < 1.:
            raise GraspFailure("bilateral_grasp_not_established")
        centre, rotation = self.box_pose()
        self.grasp_relative_positions = (n.tip_positions-centre) @ rotation
        self.grasp_relative_rotations = rotation.T @ n.tip_rotations
        self.grasp_capture = {"box_center_world_m": centre.tolist(),
                              "tool_positions_box_m": self.grasp_relative_positions.tolist(),
                              "tool_rotations_box": self.grasp_relative_rotations.tolist()}
        lift_centre = centre + [0,0,self.config.lift_m]
        self.follow_carry_path("lift", lift_centre, rotation)
        if self.recovery_stage == "carry":
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 77]))
            direction = rng.normal(size=2)
            direction /= np.linalg.norm(direction)
            magnitude = rng.uniform(self.recovery_offset_m*.5, self.recovery_offset_m)
            disturbed_centre, disturbed_rotation = self.box_pose()
            disturbed_centre[:2] += direction*magnitude
            self.follow_carry_path("carry_align", disturbed_centre, disturbed_rotation)
            measured_centre, _ = self.box_pose()
            self.recovery_start = {"stage": "carry",
                                   "target_box_centre_world_m": [0., 0., float(lift_centre[2])],
                                   "actual_box_centre_world_m": measured_centre.tolist()}
            self.start_recovery_recording()
        self.follow_carry_path("carry_align", np.array([0.,0.,lift_centre[2]]), np.eye(3))
        self.follow_carry_path("lower", np.array([0.,0.,n.control_height+BOX_SIZE[2]/2]), np.eye(3))
        release_positions, release_rotations = n.tip_positions.copy(), n.tip_rotations.copy()
        self.move_grippers("release", release_positions, release_rotations, opening)
        rim_positions = release_positions.copy()
        rim_positions[:, 2] = np.maximum(release_positions[:, 2]+.010, n.control_height+.090)
        self.move_tools("withdraw_rim", rim_positions, release_rotations, opening, speed=self.config.contact_speed_m_s)
        withdrawal_rotations = n._rotation_increment(np.array([math.radians(20.),0,0])) @ release_rotations
        self.move_tools("withdraw_orient", rim_positions, withdrawal_rotations, opening)
        above_positions = rim_positions.copy()
        above_positions[:, 2] = n.control_height+BOX_SIZE[2]+.030
        self.move_tools("withdraw_above", above_positions, withdrawal_rotations, opening)
        self.move_tools("park", np.asarray(n.initial_tip_positions), n.initial_tip_rotations, opening)
        self.stage_start("verify_released_alignment")
        n.terminate_on_alignment = True
        n.stable_steps = 0
        for _ in range(math.ceil(2./n.dt)):
            transition = self.tick(np.asarray(n.initial_tip_positions), n.initial_tip_rotations,
                                   opening, allow_complete=True)
            if transition.done:
                if transition.info["success"] and np.max(n.contact_forces) < .1:
                    return {**self.metadata(), "success": True}
                raise GraspFailure("released_alignment_failed")
        raise GraspFailure("released_alignment_timeout")

    def metadata(self):
        return {"controller": "contact_only_random_grasp_carry_release/v3", "seed": self.seed,
                "initial_box_pose_world": self.initial_box_pose,
                "recovery": self.recovery, "recovery_stage": self.recovery_stage,
                "recovery_start": self.recovery_start,
                "contact_error_height_m": self.contact_error_height_m,
                "config": asdict(self.config), "stage_frames": self.stage_counts,
                "stage_trace": self.trace, "frames": self.frames, "peak_lift_m": self.peak_lift_m,
                "max_finger_box_penetration_m": self.native.max_finger_box_penetration_m,
                "max_rim_penetration_m": self.native.max_rim_penetration_m,
                "stage_instructions": list(self.stage_instructions),
                "stage_motion": self.stage_motion,
                "sampled_grasp": {key: value.tolist() for key, value in self.sample.items()},
                "measured_grasp": getattr(self, "grasp_capture", None)}


def collect_grasp_episode(environment, writer, seed, *, config=None, max_steps=None,
                          recovery=False, recovery_offset_m=.010, recovery_stage="approach",
                          contact_error_height_m=.020):
    recorder = EpisodeRecorder(environment, buffer_dir=getattr(writer, "root", None))
    expert = GraspCarryExpert(environment, seed, config=config, step=recorder.step,
                             max_steps=max_steps, recovery=recovery,
                             recovery_offset_m=recovery_offset_m,
                             recovery_stage=recovery_stage,
                             contact_error_height_m=contact_error_height_m)
    try:
        try:
            result = expert.run()
        except GraspFailure as error:
            return {**expert.metadata(), "success": False, "saved": False, "failure_reason": str(error)}
        instructions = [expert.stage_instructions[int(value[0])]
                        for value in recorder.rows["language.stage_id"]]
        result.update(recorder.save(writer, "environment", task=instructions), saved=True)
        return result
    finally:
        recorder.close()
