"""Joint-limit-aware numerical inverse kinematics for the imported ARX tree."""

from __future__ import annotations

import numpy as np
import mujoco


class ArmIK:
    """Damped least squares on the six-dimensional world-space tip pose.

    A separate MjData is used for kinematics. Solving IK never changes the live
    physics state; the solution is sent only to the robot joint actuators.
    """

    def __init__(self, model, site_id, qids, dofs, joint_ids, rotation):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = site_id
        self.qids = qids
        self.dofs = dofs
        self.lower = model.jnt_range[joint_ids, 0] + 1e-4
        self.upper = model.jnt_range[joint_ids, 1] - 1e-4
        self.rotation = np.asarray(rotation, dtype=float)
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))
        self.quat = np.zeros(4)
        self.position_error = np.inf
        self.orientation_error = np.inf

    def _error(self, target):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        pos_error = target - self.data.site_xpos[self.site_id]
        actual = self.data.site_xmat[self.site_id].reshape(3, 3)
        relative = self.rotation @ actual.T
        mujoco.mju_mat2Quat(self.quat, relative.ravel())
        if self.quat[0] < 0:
            self.quat *= -1
        length = np.linalg.norm(self.quat[1:])
        angle = 2 * np.arctan2(length, self.quat[0])
        rotation_error = self.quat[1:] * angle / max(length, 1e-12)
        self.position_error = float(np.linalg.norm(pos_error))
        self.orientation_error = float(angle)
        return np.concatenate((pos_error, .35 * rotation_error))

    def solve(self, full_qpos, seed, target, max_iterations=30, *, rotation=None):
        if rotation is not None:
            self.rotation = np.asarray(rotation, dtype=float)
        self.data.qpos[:] = full_qpos
        self.data.qpos[self.qids] = seed
        for _ in range(max_iterations):
            error = self._error(np.asarray(target))
            if self.position_error < 1e-6 and self.orientation_error < 1e-5:
                break
            mujoco.mj_jacSite(self.model, self.data, self.jp, self.jr, self.site_id)
            jacobian = np.vstack((self.jp[:, self.dofs], .35*self.jr[:, self.dofs]))
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6)*1e-6, error)
            step *= min(1., .2 / max(np.max(np.abs(step)), 1e-12))
            self.data.qpos[self.qids] = np.clip(self.data.qpos[self.qids] + step, self.lower, self.upper)
        self._error(np.asarray(target))
        return self.data.qpos[self.qids].copy()

    @property
    def converged(self):
        return self.position_error < .0003 and self.orientation_error < .003
