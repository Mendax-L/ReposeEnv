"""pusht public environment; same API as the other push tasks."""
from __future__ import annotations
from typing import Any
import numpy as np
from common.environment import PushEnvironment, prepare_native_runtime

class PushTEnv(PushEnvironment):
    name = "pusht"
    task_description = "请推动 T 形物块，使它与目标轮廓对齐。"

    def __init__(self, render_size: int = 96) -> None:
        prepare_native_runtime()
        from .native.keypoints_env import PushTKeypointsEnv

        super().__init__(PushTKeypointsEnv(
            legacy=True,
            keypoint_visible_rate=1.0,
            agent_keypoints=False,
            render_action=False,
            render_size=render_size,
        ))

    def _native_action(self, action: np.ndarray) -> np.ndarray:
        return action

    def get_keypoints(self) -> np.ndarray:
        if self._latest_observation is None:
            raise RuntimeError("Call reset(seed) before reading keypoints.")
        return self._keypoints.copy()

    def teleop_agent(self):
        from .teleop import create_teleop
        return create_teleop(self)

    def _model_observation(self, native_obs: Any) -> np.ndarray:
        value = np.asarray(native_obs, dtype=np.float32)
        if value.shape == (40,):
            value = value[:20]
        if value.shape != (20,):
            raise ValueError(f"Expected 20 native PushT values, got {value.shape}.")
        self._keypoints = value[:18].reshape(9, 2).copy()
        return self._validate_observation(value[-2:])

