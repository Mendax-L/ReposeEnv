"""pushbox2d public environment; same API as the other push tasks."""
from __future__ import annotations
from typing import Any
import numpy as np
from common.environment import PushEnvironment, WORKSPACE_SIZE, prepare_native_runtime

class PushBox2DEnv(PushEnvironment):
    name = "pushbox2d"
    obs_dim = 12
    action_dim = 2
    action_low = (0.0,) * 2
    action_high = (WORKSPACE_SIZE,) * 2
    observation_layout = "box4_xy_two_agents_xy/v1"
    action_semantics = "attached_pair_absolute_xy"

    def __init__(self, render_size: int = 96) -> None:
        prepare_native_runtime()
        from .native.env import PushBoxEnv

        super().__init__(PushBoxEnv(obs_type="state", render_action=False, render_size=render_size))

    def _native_action(self, action: np.ndarray) -> np.ndarray:
        return action

    def get_keypoints(self) -> np.ndarray:
        if self._latest_observation is None:
            raise RuntimeError("Call reset(seed) before reading keypoints.")
        return self._latest_observation[:8].reshape(4, 2).copy()

    def teleop_agent(self):
        from .teleop import create_teleop
        return create_teleop(self)

    def _model_observation(self, native_obs: Any) -> np.ndarray:
        del native_obs
        corners = np.asarray(self.environment.get_keypoints(), dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError(f"PushBox2D keypoints must have shape (4, 2), got {corners.shape}.")
        # Four corners fully encode the rigid rectangle pose without a wrapped
        # angle field. Both agents are part of the task and the policy state.
        agent_positions = np.concatenate((
            np.asarray(self.environment.agent.position, dtype=np.float32),
            np.asarray(self.environment.internal_agent.position, dtype=np.float32),
        ))
        return self._validate_observation(np.concatenate((corners.reshape(-1), agent_positions)))

