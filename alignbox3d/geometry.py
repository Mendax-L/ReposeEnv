"""Shared-world rigid transforms; quaternions are always WXYZ."""
import math
import numpy as np
import mujoco


def rotation_vector(matrix):
    quat = quaternion(matrix)
    vector = np.zeros(3)
    mujoco.mju_quat2Vel(vector, quat, 1.)
    return vector


def quaternion(matrix):
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix, dtype=float).ravel())
    return result if result[0] >= 0 else -result


def box_rotation(roll, pitch, yaw):
    """Right-handed roll/pitch and clockwise-positive box yaw, in radians."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
            @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def smoothstep(value):
    value = float(np.clip(value, 0., 1.))
    return value**3 * (10 + value * (-15 + 6 * value))
