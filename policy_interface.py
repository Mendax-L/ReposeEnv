"""Policy interaction: local observations/actions and the TCP rollout client."""
from __future__ import annotations
import argparse
import base64
from collections import deque
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any
import numpy as np
from environment import make_environment, SUPPORTED_ENVIRONMENTS, PushEnvironment
from common.recording import _success
PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576

class PolicyInterface:
    """Select algorithm inputs without changing the recorded dataset.

    obs='state' returns the canonical state vector; 'keypoint' concatenates
    object keypoints and agent positions; 'image' returns every RGB camera stream
    plus agent positions (both cameras for AlignBox3D); 'all' returns every
    recording observation.
    """

    def __init__(self, name="pusht", *, obs="state", render_size=None):
        if obs not in ("state", "keypoint", "image", "all"):
            raise ValueError("obs must be state, keypoint, image or all.")
        self.env = make_environment(name, render_size)
        self.obs = obs

    def observe(self):
        if self.obs == "state":
            return self.env.lerobot_observation(include_images=False)["observation.state"]
        if self.obs == "keypoint":
            return np.concatenate((self.env.get_keypoints().reshape(-1), self.env.get_agent_positions()))
        if self.obs == "image":
            snapshot = self.env.lerobot_observation()
            images = {key: value.copy() for key, value in snapshot.items()
                      if key.startswith("observation.images.")}
            return {**images, "observation.agent_pos": self.env.get_agent_positions()}
        return self.env.observe()

    def reset(self, seed=0):
        self.env.reset(seed)
        return self.observe()

    def step(self, action):
        result = self.env.step(action)
        return replace(result, obs=self.observe())

    def contract(self):
        contract = self.env.contract()
        schema = self.env.recording_schema()
        if self.obs == "state":
            keys = ("observation.state",)
        elif self.obs == "keypoint":
            keys = ("observation.keypoints", "observation.agent_pos")
        elif self.obs == "image":
            keys = tuple(key for key, feature in schema["features"].items()
                         if feature["dtype"] == "video") + ("observation.agent_pos",)
        else:
            keys = tuple(key for key in schema["features"] if key.startswith("observation."))
        contract.update(observation_mode=self.obs, fps=self.env.fps,
                        input_features={key: schema["features"][key] for key in keys})
        if self.obs in ("keypoint", "state"):
            contract["input_dim"] = sum(schema["features"][key]["shape"][0] for key in keys)
        return contract

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = connection.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("Peer closed the IPC connection.")
        chunks.extend(chunk)
    return bytes(chunks)


def send_message(connection: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("IPC message exceeds 1 MiB.")
    connection.sendall(struct.pack("!I", len(data)) + data)


def receive_message(connection: socket.socket) -> dict[str, Any]:
    length = struct.unpack("!I", _receive_exact(connection, 4))[0]
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("IPC message exceeds 1 MiB.")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    payload = json.loads(_receive_exact(connection, length), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("IPC payload must be a JSON object.")
    return payload



def assert_environment_runtime(expected_env: str = "ReposeEnv") -> None:
    actual = Path(sys.prefix).name.lower()
    if actual != expected_env.lower():
        raise RuntimeError(f"ReposeEnv runtime requires Conda env '{expected_env}', got '{actual}'.")
    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("Isolation violation: PyTorch belongs in the model runtime, not ReposeEnv.")


def connect_with_retry(host: str, port: int, timeout: float) -> socket.socket:
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("Connection timeout must be positive and finite.")
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((host, port), timeout=5.0)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)


def _rgb_history(images: np.ndarray) -> dict[str, Any]:
    """Match the model server's lossless NHWC uint8 RGB wire format."""
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("RGB history must be uint8 [history, height, width, 3].")
    if 4 * ((images.size + 2) // 3) > MAX_MESSAGE_BYTES:
        raise ValueError("RGB history exceeds the 1 MiB message limit.")
    return {"shape": list(images.shape), "data": base64.b64encode(images.tobytes()).decode("ascii")}


def _finish_transitions(environment: PushEnvironment, reason: str):
    """End a 3-D episode at its current pose without advancing physics."""
    if environment.environment.phase in ("complete", "failed"):
        return
    environment.request_finish(reason)
    transition = environment.step(environment.hold_action())
    yield transition
    if not transition.done:
        raise RuntimeError("Requested episode completion did not end the episode.")


def run_rollout(
    connection: socket.socket,
    name: str,
    episodes: int,
    seed: int,
    max_steps: int,
    health: dict[str, Any],
) -> dict[str, Any]:
    """Run complete episodes; max_steps counts direct end-effector control ticks.

    Requested completion inserts no robot motion.
    """
    image_shape = health.get("image_shape")
    image_key = health.get("image_key", "observation.images.front")
    if image_shape is not None and (
        not isinstance(image_shape, (list, tuple)) or len(image_shape) != 3
        or any(type(size) is not int or size < 1 for size in image_shape) or image_shape[-1] != 3
    ):
        raise ValueError("Policy RGB shape must be [height, width, 3].")
    environment = make_environment(name, render_size=image_shape[1] if image_shape is not None else 96)
    scores: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    results: list[dict[str, Any]] = []
    spatial = environment.name == "alignbox3d"
    episode_in_progress = False
    try:
        n_obs_steps, n_action_steps = validate_policy_contract(environment, health)
        if image_shape is not None:
            feature = environment.lerobot_schema()["features"].get(image_key)
            if feature is None or feature["dtype"] != "video" or feature["shape"] != list(image_shape):
                raise RuntimeError("Policy/environment RGB camera or dimensions differ.")
            if 4 * ((n_obs_steps * int(np.prod(image_shape)) + 2) // 3) > MAX_MESSAGE_BYTES:
                raise ValueError("Policy RGB history exceeds the 1 MiB message limit.")
        for episode in range(episodes):
            episode_seed = seed + episode
            send_message(connection, {"op": "reset", "seed": episode_seed})
            response = receive_message(connection)
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "Policy reset failed."))
            observation = environment.reset(episode_seed)
            episode_in_progress = True
            history = deque((observation.copy() for _ in range(n_obs_steps)), maxlen=n_obs_steps)
            images = deque(maxlen=n_obs_steps)
            if image_shape is not None:
                frame = environment.lerobot_observation()[image_key]
                images.extend(frame.copy() for _ in range(n_obs_steps))
            score = float("-inf")
            steps = 0
            done = False
            success = False
            success_known = False
            terminated = truncated = False

            def consume(transition, *, update_inputs=True):
                nonlocal score, steps, done, success, success_known, terminated, truncated
                if update_inputs:
                    history.append(transition.obs)
                    if image_shape is not None:
                        images.append(environment.lerobot_observation()[image_key])
                if not np.isfinite(transition.reward):
                    raise ValueError("Environment reward must be finite.")
                score = max(score, transition.reward)
                current_success, known = _success(environment, transition.info)
                success = success or (current_success and known)
                success_known = success_known or known
                terminated, truncated = transition.terminated, transition.truncated
                done = bool(transition.done or terminated or truncated)
                steps += 1

            while not done:
                if steps >= max_steps:
                    if spatial:
                        for transition in _finish_transitions(environment, "max_steps"):
                            consume(transition, update_inputs=False)
                    break
                if spatial and environment.environment.phase != "control":
                    actions = environment.hold_action()[None, :]
                else:
                    observation = np.stack(history).tolist()
                    if image_shape is not None:
                        observation = {"observation.state": observation, "image": _rgb_history(np.stack(images))}
                    send_message(connection, {"op": "predict", "obs": observation})
                    response = receive_message(connection)
                    if not response.get("ok"):
                        raise RuntimeError(response.get("error", "Policy prediction failed."))
                    actions = np.asarray(response["action"], dtype=np.float32)
                    if actions.shape != (n_action_steps, environment.action_dim) or not np.isfinite(actions).all():
                        raise RuntimeError(f"Invalid policy action shape or values: {actions.shape}.")
                    if np.any(actions < environment.action_low) or np.any(actions > environment.action_high):
                        raise RuntimeError("Policy actions exceed the environment bounds.")
                for action in actions:
                    consume(environment.step(action))
                    # Stop consuming a model chunk once the episode completes.
                    if done or steps >= max_steps or (spatial and environment.environment.phase != "control"):
                        break
            episode_in_progress = False
            scores.append(score)
            lengths.append(steps)
            successes.append(success)
            results.append({"seed": episode_seed, "steps": steps, "score": score, "success": success,
                            "success_known": success_known, "terminated": terminated,
                            "truncated": truncated or (steps >= max_steps and not terminated)})
    except BaseException:
        if spatial and episode_in_progress:
            try:
                for _ in _finish_transitions(environment, "policy_error"):
                    pass
            except BaseException:
                pass
        # Completion or cleanup errors must not replace the original model,
        # connection, rendering or user-interrupt exception.
        try:
            environment.close()
        except BaseException:
            pass
        raise
    else:
        environment.close()
    return {
        "environment": name,
        "episodes": episodes,
        "seed": seed,
        "max_steps": max_steps,
        "obs_shape": [environment.obs_dim],
        "action_shape": [environment.action_dim],
        "environment_contract": environment.contract(),
        "mean_score": float(np.mean(scores)),
        "success_rate": float(np.mean(successes)),
        "scores": scores,
        "steps": lengths,
        "results": results,
    }


def validate_policy_contract(environment: PushEnvironment, health: dict[str, Any]) -> tuple[int, int]:
    native = environment.contract()
    received = health.get("contract", {})
    for key in ("obs_dim", "action_dim", "observation_layout", "action_semantics", "action_low", "action_high"):
        if received.get(key) != native[key]:
            raise RuntimeError(f"Policy/environment contract mismatch at {key}: {received.get(key)} != {native[key]}.")
    schema = environment.lerobot_schema()
    for key, feature in (("observation_names", "observation.state"), ("action_names", "action")):
        if key in received and received[key] != schema["features"][feature]["names"]:
            raise RuntimeError(f"Policy/environment field order differs at {key}.")
    if "fps" in received and received["fps"] != environment.fps:
        raise RuntimeError("Policy/environment sampling frequency differs.")
    obs_shape, action_shape = health.get("obs_shape", []), health.get("action_shape", [])
    if len(obs_shape) != 2 or len(action_shape) != 2:
        raise RuntimeError("Policy shapes must contain history/chunk length and channel dimension.")
    if obs_shape[1] != environment.obs_dim or action_shape[1] != environment.action_dim:
        raise RuntimeError("Policy shape dimensions differ from its contract.")
    if any(type(value) is not int or value < 1 for value in (obs_shape[0], action_shape[0])):
        raise RuntimeError("Policy history/chunk lengths must be positive integers.")
    return obs_shape[0], action_shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=SUPPORTED_ENVIRONMENTS, default="pusht")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--socket-timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/policy/metrics.json"))
    parser.add_argument("--export-contract", type=Path, help="Export feature and action contracts without a policy connection.")
    parser.add_argument("--obs", choices=("state", "keypoint", "image", "all"),
                        help="Contract input mode, or an expected TCP input mode (otherwise negotiated with the model).")
    args = parser.parse_args()
    if min(args.episodes, args.max_steps) < 1 or not 0 <= args.seed < 2**63 - args.episodes:
        parser.error("episodes/max-steps must be positive; seeds must fit [0, 2**63)")
    if not np.isfinite(args.socket_timeout) or args.socket_timeout <= 0:
        parser.error("socket-timeout must be positive and finite")
    if args.export_contract:
        with PolicyInterface(args.env, obs=args.obs or "state") as interface:
            args.export_contract.parent.mkdir(parents=True, exist_ok=True)
            args.export_contract.write_text(json.dumps(interface.contract(), indent=2) + "\n", encoding="utf-8")
        print(f"Contract: {args.export_contract.resolve()}")
        return
    if args.obs in ("keypoint", "all"):
        parser.error("Use PolicyInterface in Python for keypoint/all; TCP negotiates state or image from the model")
    assert_environment_runtime()
    with connect_with_retry(args.host, args.port, args.connect_timeout) as connection:
        connection.settimeout(args.socket_timeout)
        send_message(connection, {"op": "health", "protocol": PROTOCOL_VERSION})
        health = receive_message(connection)
        if health.get("protocol") != PROTOCOL_VERSION or health.get("env", "").lower() == Path(sys.prefix).name.lower():
            raise RuntimeError("Protocol mismatch or model/environment runtime isolation violation.")
        try:
            negotiated_mode = "image" if health.get("image_shape") is not None else "state"
            if args.obs is not None and args.obs != negotiated_mode:
                raise RuntimeError(f"Requested {args.obs} input, but the model requires {negotiated_mode}.")
            report = run_rollout(connection, args.env, args.episodes, args.seed, args.max_steps, health)
        finally:
            try:
                send_message(connection, {"op": "close"})
                receive_message(connection)
            except (OSError, ValueError):
                pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()


