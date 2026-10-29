"""Human command generators, independent of recording and window libraries."""
from __future__ import annotations

import numpy as np


class PlanarTeleop:
    def __init__(self, environment):
        self.environment = environment

    def act(self, observation=None, *, mouse=None, direction=(0, 0, 0)):
        """Absolute XY target for PushT's agent."""
        native = self.environment.environment
        position = native.agent.position
        action = (mouse if mouse is not None else
                  np.asarray(position) + np.asarray(direction[:2]) * 60 / self.environment.fps)
        return np.clip(action, self.environment.action_low, self.environment.action_high).astype(np.float32)


class SpatialTeleop:
    def __init__(self, environment):
        self.environment = environment

    def act(self, observation=None, *, mouse=None, direction=(0, 0), yaw=0.,
            rotation=(0., 0., 0.), arm="both", gripper=None):
        """Tool XYZ and right-handed world rotation-vector commands.

        arm selects inner, outer or both. The clockwise yaw shortcut
        rotates each selected tool about its own origin for keyboard use.
        gripper is an absolute opening ratio 0..1; None retains each target.
        """
        direction = np.asarray(direction, dtype=np.float32)
        rotation = np.asarray(rotation, dtype=np.float32)
        if (direction.shape not in ((2,), (3,), (14,)) or not np.isfinite(direction).all()
                or rotation.shape != (3,) or not np.isfinite(rotation).all()
                or not np.isfinite(yaw) or arm not in ("inner", "outer", "both")):
            raise ValueError("Control requires finite XYZ/rotation and arm inner, outer or both.")
        native = self.environment.environment
        if native.phase != "control":
            return native.hold_action()
        if direction.shape == (14,):
            if yaw or np.any(rotation) or gripper is not None:
                raise ValueError("A full 14-D action cannot be combined with rotation or gripper arguments.")
            return np.clip(direction, self.environment.action_low, self.environment.action_high)
        translation = np.zeros(3)
        translation[:len(direction)] = np.clip(direction, -1, 1)
        action = native.hold_action().reshape(2, 7)
        indices = [0, 1] if arm == "both" else [0 if arm == "inner" else 1]
        action[indices, :3] = translation
        action[indices, 3:6] = np.clip(rotation, -1, 1)
        action[indices, 5] -= yaw
        if gripper is not None:
            opening = np.asarray(gripper, dtype=np.float32)
            if opening.shape not in ((), (2,)) or not np.isfinite(opening).all():
                raise ValueError("gripper must be a finite opening ratio or inner/outer pair")
            values = np.full(2, opening.item()) if opening.shape == () else opening
            action[indices, 6] = np.clip(values[indices], 0, 1)
        return np.clip(action.ravel(), self.environment.action_low, self.environment.action_high)
