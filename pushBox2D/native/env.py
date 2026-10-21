# Adapted from gym-pusht (https://github.com/huggingface/gym-pusht), Apache-2.0,
# itself derived from Diffusion Policy (https://github.com/real-stanford/diffusion_policy), MIT.
#
# Modifications: the "tee" block is replaced by a single rectangular "box" block, so that the task
# is PushBox instead of PushT. Two attached edge grips share one XY target;
# coverage rewards, rendering and observation layout retain the PushT conventions.

import collections
import os
import warnings

import cv2
import numpy as np

import gymnasium as gym

with warnings.catch_warnings():
    # Filter out DeprecationWarnings raised from pkg_resources
    warnings.filterwarnings("ignore", "pkg_resources is deprecated as an API", category=DeprecationWarning)
    import pygame

import pymunk
import pymunk.pygame_util
import shapely.geometry as sg
spaces = gym.spaces
from pymunk.vec2d import Vec2d

from .rendering import DrawOptions

RENDER_MODES = ["rgb_array"]
if os.environ.get("MUJOCO_GL") != "egl":
    RENDER_MODES.append("human")


def _create_box_vertex_order(size=(1, 2)):
    """The order in which `pymunk.Poly.create_box` returns the corners of a box.

    Returned as indices into the canonical corner list
    `[(-w/2, -h/2), (-w/2, h/2), (w/2, h/2), (w/2, -h/2)]`. The keypoint observation is indexed by this
    order, so it is probed once from pymunk rather than hardcoded: it is a chipmunk implementation
    detail that could differ between the pymunk versions this package supports (>=6.6).
    """
    canonical = [(-0.5, -1.0), (-0.5, 1.0), (0.5, 1.0), (0.5, -1.0)]
    probe = [tuple(v) for v in pymunk.Poly.create_box(None, size).get_vertices()]
    return tuple(canonical.index(v) for v in probe)


_BOX_VERTEX_ORDER = _create_box_vertex_order()


def box_local_vertices(size):
    """The four corners of a box of the given size, in local coordinates.

    Same order as `pymunk.Poly.create_box`, and therefore the same order as the keypoints the block has
    always reported, so that geometry computed from `block_size` matches geometry read off a box shape.
    """
    w, h = size
    canonical = [(-w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2)]
    return [Vec2d(*canonical[i]) for i in _BOX_VERTEX_ORDER]


class PushBoxEnv(gym.Env):
    """
    ## Description

    PushBox environment.

    The goal of the agent is to push the block to the goal zone. The agent is a circle and the block is a
    rectangle (a "box"). This is the PushT task of gym-pusht with the tee-shaped block replaced by a box
    block, and is meant as a drop-in replacement for control / imitation-learning policy experiments.

    Blue and red markers track attached points on the left and right edges in the lower half of the box.
    Their circles are visual markers, not collision geometry; only the box collides with the workspace.

    ## Action Space

    One continuous [x, y] vector in [0, 512] targets the midpoint of both grips.
    Both grips move together through the same PD controller; pivot joints keep the box attached.
    Targets are clipped together to keep the box and both grips inside the workspace walls.

    ## Observation Space

    If `obs_type` is set to `state`, the observation is the 7-dimensional vector
    [agent_x, agent_y, internal_agent_x, internal_agent_y, block_x, block_y, block_angle] (7,).

    If `obs_type` is set to `environment_state_agent_pos` the observation space is a dictionary with:
    - `environment_state`: 8-dimensional vector representing the keypoint locations of the box (in [x0, y0, x1, y1, ...]
        format). The values are in the range [0, 512]. See `get_keypoints` for a diagram showing the location of the
        keypoint indices. Note: the tee of gym-pusht has two rectangles and therefore 16 keypoint values, the box has
        a single rectangle and therefore 8.
    - `agent_pos`: A 4-dimensional vector `[external_x, external_y, internal_x, internal_y]`.

    If `obs_type` is set to `pixels`, the observation is a `render_size` square RGB image.

    ## Rewards

    The reward is the coverage of the block in the goal zone. The reward is 1.0 if the block is fully in the goal zone.

    ## Success Criteria

    The environment is considered solved if the block is at least 95% in the goal zone.

    ## Starting State

    The upright goal has its short edge parallel to the screen's horizontal axis. The block starts with a
    small target-relative position error. The two grips start attached at local (+/-width/2, height/4).
    The initial angle matches the goal because the shared XY action controls translation only.

    ## Episode Termination

    The episode terminates when the block is at least 95% in the goal zone.

    ## Arguments

    The leading constructor arguments and the `reset`, `step`, and `render` return/call conventions match
    `PushTEnv` in this repository.

    * `obs_type`: (str) The observation type. Can be either `state`, `environment_state_agent_pos`, `pixels` or
      `pixels_agent_pos`. Default is `state`.

    * `block_size`: (tuple) The (width, height) of the box in pixels. Default is `(125, 250)`. A non-square box makes
      the orientation of the block matter, which is what makes the task non-trivial. A square box makes every
      quarter-turn equivalent.

    * `block_wall_thickness`: (float) Thickness of the four walls of the hollow block, in pixels. Default is
      `10.0`. The mass, inertia, outer outline, goal zone and success criterion use the full `block_size`.

    * `success_threshold`: (float) Fraction of the goal zone the block must cover to count as pushed
      onto it, in `(0, 1]`. Default is `0.95`, as in gym-pusht. Lowering it makes the task attainable
      for policies that align the block within a few degrees but do not hit the exact goal pose.

    * `block_cog`: (tuple) The center of gravity of the block if different from the center of mass. Default is `None`.

    * `damping`: (float) The damping factor of the environment if different from 0. Default is `None`.

    * `block_spawn_offset`: (tuple) Maximum target-relative x/y offset. Default is `(10, 25)` pixels.

    * `block_spawn_angle`: (float) Must be `0`: shared XY control preserves orientation.

    ## Reset Arguments

    Passing `reset_to_state` to the constructor resets the environment to a specific state. The state
    must contain exactly 7 values in `obs_type="state"` order.

    ```python
    >>> env = PushBoxEnv(reset_to_state=[193.5, 262.5, 318.5, 262.5, 256, 200, 0])
    >>> state = env.reset()
    >>> state
    array([193.5, 262.5, 318.5, 262.5, 256., 200., 0.])
    ```

    ## Version History

    * v0: Original version

    ## References

    * gym-pusht: https://github.com/huggingface/gym-pusht
    * Diffusion Policy: https://github.com/real-stanford/diffusion_policy
    """

    metadata = {
        "render.modes": RENDER_MODES,
        "video.frames_per_second": 10,
        # Gymnasium aliases; harmless for classic Gym callers.
        "render_modes": RENDER_MODES,
        "render_fps": 10,
    }
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=True,
        render_size=96,
        reset_to_state=None,
        *,
        obs_type="state",
        block_size=(125, 250),
        block_wall_thickness=10.0,
        success_threshold=0.95,
        block_spawn_offset=(10.0, 25.0),
        block_spawn_angle=0.0,
    ):
        super().__init__()
        self._seed = None
        self.seed()
        self.window_size = 512
        self.render_size = render_size
        self.sim_hz = 100
        self.legacy = legacy
        self.render_action = render_action
        self.reset_to_state = reset_to_state

        # Observations
        self.obs_type = obs_type

        # Block geometry
        self.block_size = tuple(block_size)
        self.n_keypoints = 4  # one rectangle, 4 vertices
        self.agent_radius = 15

        # Keep the original hollow frame geometry.
        self.block_wall_thickness = float(block_wall_thickness)
        if not 0 < self.block_wall_thickness < min(self.block_size) / 2 - self.agent_radius:
            raise ValueError(
                f"block_wall_thickness={block_wall_thickness} leaves a cavity too small for the internal agent "
                f"(radius {self.agent_radius}) in a {self.block_size} block"
            )

        if len(block_spawn_offset) != 2 or min(block_spawn_offset) < 0:
            raise ValueError(f"block_spawn_offset must contain two non-negative values, got {block_spawn_offset}")
        if block_spawn_angle != 0:
            raise ValueError("block_spawn_angle must be 0 because attached XY control preserves orientation")
        self.block_spawn_offset = np.asarray(block_spawn_offset, dtype=np.float64)
        self.block_spawn_angle = float(block_spawn_angle)

        # Initialize spaces
        self._initialize_observation_space()

        # Physics
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["video.frames_per_second"]
        self.dt = 0.01
        self.block_cog = block_cog
        self.damping = damping

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

        # Off-screen surface the environment is drawn on.
        self.screen = None

        self.teleop = None
        self.latest_action = None

        # Coverage above which the block counts as pushed onto the goal. gym-pusht uses 0.95; the
        # human demos in `data/demo_dual.zarr` top out at 0.950, so RL runs that want an attainable
        # target pass something a little below (see `--success-threshold` in collect_rl.py).
        if not 0.0 < success_threshold <= 1.0:
            raise ValueError(f"success_threshold must be in (0, 1], got {success_threshold}")
        self.success_threshold = float(success_threshold)

    def _initialize_observation_space(self):
        if self.obs_type == "state":
            # [external_x, external_y, internal_x, internal_y, block_x, block_y, block_angle]
            self.observation_space = spaces.Box(
                low=np.zeros(7),
                high=np.array([512] * 6 + [2 * np.pi]),
                dtype=np.float64,
            )
        elif self.obs_type == "environment_state_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "environment_state": spaces.Box(
                        low=np.zeros(2 * self.n_keypoints),
                        high=np.full((2 * self.n_keypoints,), 512),
                        dtype=np.float64,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.zeros(4),
                        high=np.full(4, 512),
                        dtype=np.float64,
                    ),
                },
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.render_size, self.render_size, 3), dtype=np.uint8
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.render_size, self.render_size, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=np.zeros(4),
                        high=np.full(4, 512),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, environment_state_agent_pos, "
                "pixels_agent_pos]"
            )

        self.action_space = spaces.Box(low=0, high=512, shape=(2,), dtype=np.float64)

    def _get_coverage(self):
        # Coverage uses the hollow frame's outer outline, not its wall area.
        goal_geom = self.get_block_geom(self.get_goal_pose_body(self.goal_pose))
        block_geom = self.get_block_geom(self.block)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        return intersection_area / goal_area

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite shared XY target with shape (2,)")
        self.n_contact_points = 0
        n_steps = int(1 / (self.dt * self.control_hz))
        center = self.grip_center
        grip_offsets = np.asarray([self.agent.position, self.internal_agent.position]) - center
        box_offsets = np.asarray(self.get_block_polygon(self.block)[:-1]) - center
        lower = np.minimum(box_offsets.min(axis=0), grip_offsets.min(axis=0) - self.agent_radius)
        upper = np.maximum(box_offsets.max(axis=0), grip_offsets.max(axis=0) + self.agent_radius)
        target = np.clip(action, 7.0 - lower, 504.0 - upper)
        self.latest_action = target.copy()

        for _ in range(n_steps):
            velocity = (np.asarray(self.agent.velocity) + np.asarray(self.internal_agent.velocity)) / 2
            velocity += (self.k_p * (target - self.grip_center) - self.k_v * velocity) * self.dt
            self.agent.velocity = tuple(velocity)
            self.internal_agent.velocity = tuple(velocity)
            self.space.step(self.dt)

        # Compute reward
        coverage = self._get_coverage()
        reward = np.clip(coverage / self.success_threshold, 0.0, 1.0)
        terminated = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()
        info["coverage"] = coverage

        return observation, reward, terminated, info

    def reset(self):
        # PushT recreates its seeded RandomState on reset, so the same seed
        # reproduces the same episode rather than advancing a global RNG.
        self.np_random = np.random.default_rng(self._seed)
        self._setup()

        if self.reset_to_state is not None:
            state = np.asarray(self.reset_to_state, dtype=np.float64)
            if state.shape != (7,):
                raise ValueError(f"reset_to_state must have shape (7,), got {state.shape}")
        else:
            block_x, block_y, block_angle = self._sample_block_pose()
            agent_x, agent_y, internal_x, internal_y = self._sample_agent_positions(
                (block_x, block_y, block_angle)
            )
            state = np.array([agent_x, agent_y, internal_x, internal_y, block_x, block_y, block_angle])
        self._set_state(state)

        return self._get_obs()

    def _sample_block_pose(self):
        """Sample a small pose error around the upright target pose."""
        position = self.goal_pose[:2] + self.np_random.uniform(
            -self.block_spawn_offset, self.block_spawn_offset
        )
        angle = self.goal_pose[2] + self.np_random.uniform(-self.block_spawn_angle, self.block_spawn_angle)
        return position[0], position[1], angle

    @property
    def grip_center(self):
        """World XY midpoint commanded by the shared action."""
        return (np.asarray(self.agent.position) + np.asarray(self.internal_agent.position)) / 2

    def _sample_agent_positions(self, block_pose):
        """Place both grips directly on the box's lower side edges."""
        width, height = self.block_size
        block_body = self.get_goal_pose_body(np.asarray(block_pose))
        external = block_body.local_to_world((-width / 2, height / 4))
        internal = block_body.local_to_world((width / 2, height / 4))
        return external.x, external.y, internal.x, internal.y

    def _draw(self):
        # Create a screen
        screen = pygame.Surface((512, 512))
        screen.fill((255, 255, 255))
        draw_options = DrawOptions(screen)
        draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

        # Draw goal pose
        goal_body = self.get_goal_pose_body(self.goal_pose)
        goal_polygon = self.get_block_polygon(goal_body)
        goal_polygon = [pymunk.pygame_util.to_pygame(point, screen) for point in goal_polygon]
        pygame.draw.polygon(screen, pygame.Color("LightGreen"), goal_polygon)

        # Draw physics geometry first.
        self.space.debug_draw(draw_options)
        # Both grip markers use the same final layer, above every box wall.
        for body, color in ((self.agent, "RoyalBlue"), (self.internal_agent, "Red")):
            position = pymunk.pygame_util.to_pygame(body.position, screen)
            pygame.draw.circle(screen, pygame.Color(color), position, self.agent_radius)
        self.screen = screen
        return screen

    def _get_img(self, screen, width, height, render_action=False):
        img = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
        img = cv2.resize(img, (width, height))
        render_size = min(width, height)
        if render_action and self.latest_action is not None:
            targets = [(self.latest_action, (0, 0, 255))]
            marker_size = int(8 / 96 * render_size)
            thickness = int(1 / 96 * render_size)
            for target, color in targets:
                if target is None:
                    continue
                coord = (np.array(target) / 512 * [height, width]).astype(np.int32)
                cv2.drawMarker(
                    img,
                    coord,
                    color=color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img

    def render(self, mode):
        return self._render_frame(mode)

    def _render_frame(self, mode):
        screen = self._draw()  # draw the environment on a screen

        if mode == "rgb_array":
            return self._get_img(
                screen,
                width=self.render_size,
                height=self.render_size,
                render_action=self.render_action,
            )
        elif mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((512, 512))
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(
                screen, screen.get_rect()
            )  # copy our drawings from `screen` to the visible window
            pygame.event.pump()
            # One frame is drawn per env step, so tick at the control rate to play back in real time. gym-pusht
            # ticks at `render_fps * n_substeps` (100 fps), which plays 10x too fast and floods X11 forwarding.
            self.clock.tick(self.metadata["video.frames_per_second"])
            pygame.display.update()
        else:
            raise ValueError(mode)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def teleop_agent(self):
        teleop_agent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            keys = pygame.key.get_pressed()
            direction = np.array([
                int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT]),
                int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP]),
            ])
            return self.grip_center + direction * 60 / self.control_hz

        return teleop_agent(act)

    def _get_obs(self):
        if self.obs_type == "state":
            agent_position = np.array(self.agent.position)
            internal_agent_position = np.array(self.internal_agent.position)
            block_position = np.array(self.block.position)
            block_angle = self.block.angle % (2 * np.pi)
            return np.concatenate(
                [agent_position, internal_agent_position, block_position, [block_angle]], dtype=np.float64
            )

        if self.obs_type == "environment_state_agent_pos":
            return {
                "environment_state": self.get_keypoints().flatten(),
                "agent_pos": np.concatenate(
                    [np.array(self.agent.position), np.array(self.internal_agent.position)]
                ),
            }

        pixels = self._render_frame("rgb_array")
        if self.obs_type == "pixels":
            return pixels
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": np.concatenate(
                    [np.array(self.agent.position), np.array(self.internal_agent.position)]
                ),
            }

    # Kept as a small compatibility alias for callers of the previous version.
    def get_obs(self):
        return self._get_obs()

    def get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, self.block_size)
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def get_block_polygon(self, body):
        """World coordinates of the block's outer outline, as a closed list of vertices.

        Derived from `block_size` and the body transform rather than from the four wall shapes.
        """
        points = [body.local_to_world(v) for v in box_local_vertices(self.block_size)]
        points += [points[0]]
        return points

    def get_block_inner_polygon(self, body):
        """World coordinates of the block's inner cavity outline."""
        t = self.block_wall_thickness
        size = (self.block_size[0] - 2 * t, self.block_size[1] - 2 * t)
        points = [body.local_to_world(v) for v in box_local_vertices(size)]
        points += [points[0]]
        return points

    def get_block_geom(self, body):
        """Shapely polygon of the block's outer outline for the given body transform.

        This is the geometry the task is defined on: coverage, the goal zone and the agent spawn band all
        use the full block footprint, not the walls of the hollow frame.
        """
        return sg.Polygon([(v.x, v.y) for v in self.get_block_polygon(body)[:-1]])

    def _get_info(self):
        n_steps = int(1 / (self.dt * self.control_hz))
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(list(self.block.position) + [self.block.angle]),
            "goal_pose": np.array(self.goal_pose),
            "n_contacts": n_contact_points_per_step,
        }
        info["pos_internal_agent"] = np.array(self.internal_agent.position)
        info["vel_internal_agent"] = np.array(self.internal_agent.velocity)
        return info

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = self.damping if self.damping is not None else 0.0
        self.teleop = False
        self.latest_action = None
        self.grip_joints = []

        # Add walls
        walls = [
            self.add_segment(self.space, (5, 506), (5, 5), 2),
            self.add_segment(self.space, (5, 5), (506, 5), 2),
            self.add_segment(self.space, (506, 5), (506, 506), 2),
            self.add_segment(self.space, (5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone
        self.agent = self.add_circle(self.space, (256, 400), self.agent_radius)
        self.block, self._block_shapes = self.add_hollow_box(
            self.space, (256, 300), 0, size=self.block_size, thickness=self.block_wall_thickness
        )
        self.internal_agent = self.add_circle(self.space, (256, 300), self.agent_radius, color="Red")
        # Upright to a human viewer: the rectangle's short edge is horizontal.
        self.goal_pose = np.array([256, 200, 0.0])  # x, y, theta (in radians)
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog

        # Add collision handling. pymunk 7 replaced `Space.add_collision_handler` with `Space.on_collision`,
        # both are supported here so that the environment runs on pymunk >=6.6 and >=7.
        if hasattr(self.space, "on_collision"):
            self.collision_handeler = self.space.on_collision(0, 0, post_solve=self._handle_collision)
        else:
            self.collision_handeler = self.space.add_collision_handler(0, 0)
            self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _set_state(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(f"state must be finite with shape (7,), got {state.shape}")
        if abs(np.arctan2(np.sin(state[6]), np.cos(state[6]))) > 1e-8:
            raise ValueError("reset angle must be 0 because attached XY control preserves orientation")
        expected_grips = self._sample_agent_positions(state[4:7])
        if not np.allclose(state[:4], expected_grips, atol=1e-6, rtol=0):
            raise ValueError("reset grip positions must match the attached left/right box edges")
        if self.grip_joints:
            self.space.remove(*self.grip_joints)
        self.agent.position = list(state[:2])
        self.internal_agent.position = list(state[2:4])
        self.agent.velocity = self.internal_agent.velocity = (0, 0)
        # Setting angle rotates with respect to center of mass, therefore will modify the geometric position if not
        # the same as CoM. Therefore should theoretically set the angle first. But for compatibility with legacy data,
        # we do the opposite.
        if self.legacy:
            self.block.position = list(state[4:6])
            self.block.angle = state[6]
        else:
            self.block.angle = state[6]
            self.block.position = list(state[4:6])

        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0
        width, height = self.block_size
        self.grip_joints = [
            pymunk.PivotJoint(self.block, self.agent, (-width / 2, height / 4), (0, 0)),
            pymunk.PivotJoint(self.block, self.internal_agent, (width / 2, height / 4), (0, 0)),
        ]
        for joint in self.grip_joints:
            joint.collide_bodies = False
            joint.error_bias = (1 - 0.5) ** 60
        self.space.add(*self.grip_joints)
        self.space.step(self.dt)

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = int(seed)
        self.np_random = np.random.default_rng(self._seed)

    @staticmethod
    def add_segment(space, a, b, radius):
        # TODO(rcadene): rename add_segment to make_segment, since it is not added to the space
        shape = pymunk.Segment(space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")  # https://htmlcolorcodes.com/color-names
        return shape

    @staticmethod
    def add_circle(space, position, radius, color="RoyalBlue"):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color(color)
        # The radius is for display only: attached points cannot push or collide.
        shape.filter = pymunk.ShapeFilter(mask=0)
        space.add(body, shape)
        return body

    @staticmethod
    def add_hollow_box(space, position, angle, size=(125, 250), thickness=10, color="LightSlateGray", mask=None):
        """Rectangle frame (a "回" shape): one body carrying four wall shapes with an empty cavity.

        The body keeps the mass and inertia of the solid box, and the walls sit on its outline, so
        anything pushing the block from the outside gets the same block it always did. What the frame
        buys is the cavity: the internal agent can sit inside the block and push the walls from within.
        """
        if mask is None:
            mask = pymunk.ShapeFilter.ALL_MASKS()
        w, h = size
        if not 0 < thickness < min(w, h) / 2:
            raise ValueError(f"thickness={thickness} leaves no cavity in a {w}x{h} block")
        mass = 1
        inertia = pymunk.moment_for_box(mass, size)
        body = pymunk.Body(mass, inertia)
        body.angle = angle
        body.position = position
        body.friction = 1

        # (centre, size) of the four walls, in body-local coordinates
        t = thickness
        wall_layout = [
            ((0, h / 2 - t / 2), (w, t)),
            ((0, -(h / 2 - t / 2)), (w, t)),
            ((-(w / 2 - t / 2), 0), (t, h - 2 * t)),
            ((w / 2 - t / 2, 0), (t, h - 2 * t)),
        ]
        shapes = []
        for (cx, cy), wall_size in wall_layout:
            verts = [Vec2d(cx + v.x, cy + v.y) for v in box_local_vertices(wall_size)]
            shape = pymunk.Poly(body, verts)
            shape.color = pygame.Color(color)
            shape.filter = pymunk.ShapeFilter(mask=mask)
            shapes.append(shape)
        space.add(body, *shapes)
        return body, shapes

    def get_keypoints(self):
        """Get a (4, 2) numpy array with the block's outer keypoints, in `pymunk_override` drawing order.

        The hollow frame reports the four corners of its outer rectangular outline.

        0───────────1
        │           │
        │           │
        │           │
        3───────────2
        """
        return np.vstack([np.array(p) for p in self.get_block_polygon(self.block)[:-1]])
