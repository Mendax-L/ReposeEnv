"""Public simulator API: make_environment(name), reset, step, observe, render.

All policy-independent observations are available through observe(). Selecting
keypoints or RGB belongs to policy_interface.py, never to the recorder.
"""
from __future__ import annotations

from common.environment import PushEnvironment, StepResult
from common.schema import SUPPORTED_ENVIRONMENTS


def make_environment(name: str, render_size: int | None = None) -> PushEnvironment:
    name = name.lower()
    render_size = (512 if name == "pushbox3d" else 96) if render_size is None else render_size
    if type(render_size) is not int or render_size < 16 or render_size % 2:
        raise ValueError("render_size must be an even integer >= 16.")
    if name == "pusht":
        from pushT.env import PushTEnv
        return PushTEnv(render_size)
    if name == "pushbox2d":
        from pushBox2D.env import PushBox2DEnv
        return PushBox2DEnv(render_size)
    if name == "pushbox3d":
        from pushBox3D.env import PushBox3DEnv
        return PushBox3DEnv(render_size)
    raise ValueError(f"Unknown environment {name!r}; choose from {SUPPORTED_ENVIRONMENTS}.")


__all__ = ["make_environment", "PushEnvironment", "StepResult", "SUPPORTED_ENVIRONMENTS"]
