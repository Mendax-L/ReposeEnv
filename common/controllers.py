"""Human command generators, independent of recording and window libraries."""
from __future__ import annotations

import numpy as np


class PlanarTeleop:
    def __init__(self, environment):
        self.environment = environment

    def act(self, observation=None, *, mouse=None, direction=(0, 0, 0)):
        """Absolute XY target for the attached grip center, or PushT's agent."""
        native = self.environment.environment
        position = native.grip_center if self.environment.name == "pushbox2d" else native.agent.position
        action = (mouse if self.environment.name == "pusht" and mouse is not None else
                  np.asarray(position) + np.asarray(direction[:2]) * 60 / self.environment.fps)
        return np.clip(action, self.environment.action_low, self.environment.action_high).astype(np.float32)


class SpatialTeleop:
    def __init__(self, environment):
        self.environment = environment

    def act(self, observation=None, *, mouse=None, direction=(0, 0), yaw=0.):
        """Centre XY plus clockwise-positive yaw; vertical motion is automatic."""
        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape not in ((2,), (3,)) or not np.isfinite(direction).all() or not np.isfinite(yaw):
            raise ValueError("Control requires finite XY and yaw values.")
        if direction.shape == (3,) and yaw != 0:
            raise ValueError("Supply yaw in direction[2] or the yaw argument, not both.")
        action = direction if direction.shape == (3,) else np.asarray([*direction, yaw], dtype=np.float32)
        if self.environment.environment.phase != "control":
            return np.zeros(3, dtype=np.float32)
        return np.clip(action, -1, 1).astype(np.float32)
