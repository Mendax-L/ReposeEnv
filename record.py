"""Record human, supplied or scripted straight-line actions as LeRobot v3.0."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from common.recording import recording_schema, record_episode, record_straight_episode, load_actions
from common.teleoperation import DemonstrationSession, run_window
from environment import make_environment, SUPPORTED_ENVIRONMENTS

TASKS = {
    "pusht": "Push the T into its goal",
    "pushbox2d": "Move both attached grips together to align the hollow box",
    "pushbox3d": "Move both attached ARX L5 grips together to align the upper box with the lower box",
}


def write_collection_report(output, episodes, requested, elapsed):
    """Keep generated trajectory results separate from human demonstrations."""
    successes = sum(bool(episode["success"]) for episode in episodes)
    report = {"kind": "scripted_straight_line", "human_demonstration": False,
              "yaw_convention": "clockwise_positive_viewed_from_above",
              "requested_episodes": requested, "saved_episodes": len(episodes),
              "successful_episodes": successes,
              "success_rate": successes / len(episodes) if episodes else 0.0,
              "elapsed_seconds": elapsed, "episodes": episodes}
    path = output / "meta/collection.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return report


def main(default_env="pusht"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=SUPPORTED_ENVIRONMENTS, default=default_env)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--episodes", type=int, help="Human: 0/unset means unlimited. Straight-line: default 100.")
    parser.add_argument("--render-size", type=int, help="Recorded camera width; default 512 for PushBox3D, 96 for 2-D.")
    parser.add_argument("--speed-mm-s", type=float, default=20.0, help="Straight-line shared XY speed in mm/s; default 20.")
    parser.add_argument("--yaw-speed-deg-s", type=float, default=10.0, help="Straight-line centre yaw speed in degrees/s; default 10.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--actions", type=Path, help="Replay one episode of controller .npy/.npz actions instead of human control.")
    source.add_argument("--validate", type=Path, metavar="DATASET", help="Check an existing LeRobot dataset without starting a simulator.")
    source.add_argument("--straight-line", action="store_true", help="Generate scripted PushBox3D straight trajectories; these are not human demonstrations.")
    args = parser.parse_args()
    if args.validate:
        from common.lerobot import validate_dataset
        print(json.dumps(validate_dataset(args.validate), indent=2))
        return
    args.episodes = (100 if args.straight_line else 0) if args.episodes is None else args.episodes
    args.render_size = (512 if args.env == "pushbox3d" else 96) if args.render_size is None else args.render_size
    max_steps = args.max_steps if args.max_steps is not None else (6000 if args.env == "pushbox3d" else 3000)
    if max_steps < 1 or args.episodes < 0 or not 0 <= args.seed < 2**63:
        parser.error("max-steps must be positive; episodes nonnegative; seed in [0, 2**63)")
    if args.render_size < 16 or args.render_size % 2:
        parser.error("render-size must be an even integer >= 16")
    if args.actions and args.episodes not in (0, 1):
        parser.error("--actions supplies exactly one episode")
    if args.straight_line:
        if args.env != "pushbox3d":
            parser.error("--straight-line currently supports --env pushbox3d")
        if args.episodes < 1 or args.seed + args.episodes > 2**63:
            parser.error("straight-line episodes must be positive and all episode seeds must fit [0, 2**63)")
        if not math.isfinite(args.speed_mm_s) or not 0 < args.speed_mm_s <= 50:
            parser.error("straight-line speed-mm-s must be finite and within (0, 50]")
        if not math.isfinite(args.yaw_speed_deg_s) or not 0 < args.yaw_speed_deg_s <= 25:
            parser.error("straight-line yaw-speed-deg-s must be finite and within (0, 25]")
    task = args.task if args.task is not None else TASKS[args.env]
    if not task.strip():
        parser.error("task must contain a task description")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    kind = "scripted-straight-yaw-cw" if args.straight_line else "replay" if args.actions else "human"
    output = args.output or Path(__file__).parent / "training_data/lerobot" / args.env / f"{kind}-{stamp}"
    actions = load_actions(args.actions) if args.actions else None
    from common.lerobot import LeRobotWriter, validate_dataset
    environment = make_environment(args.env, render_size=args.render_size)
    if hasattr(environment.environment, "render_action"):
        environment.environment.render_action = False
    session = None
    writer = None
    replay_saved = False
    generated = []
    try:
        if actions is not None and actions.shape[1] != environment.action_dim:
            parser.error(f"{args.env} requires {environment.action_dim} action channels")
        environment.reset(args.seed)
        provenance = {
            "kind": "scripted_straight_line" if args.straight_line else "supplied_controller_actions" if args.actions else "human_teleoperation",
            "human_demonstration": not (args.straight_line or args.actions),
            "environment": args.env, "initial_seed": args.seed,
            "episode_seed_rule": "initial_seed + saved episode_index; discarded attempts reuse the seed",
            "modalities": "all available native observations and RGB",
            "observation_selection": "deferred to the training algorithm",
            "max_steps": max_steps, "demonstration_quality": "success is recorded, not assumed",
            "rgb_action_markers": False,
            "render_size": args.render_size,
            "environment_contract": environment.contract(),
            "initialization": environment.lerobot_schema().get("initialization"),
        }
        if args.actions:
            provenance.update(action_source=str(args.actions.resolve()),
                              action_source_sha256=hashlib.sha256(args.actions.read_bytes()).hexdigest())
        if args.straight_line:
            provenance.update(episodes_requested=args.episodes, controller={
                "name": "straight_line", "speed_metres_per_second": args.speed_mm_s / 1000,
                "yaw_speed_radians_per_second": math.radians(args.yaw_speed_deg_s),
                "target_box_xy_metres": [0.0, 0.0],
                "target_box_yaw_radians": 0.0,
                "yaw_positive_direction": "clockwise viewed from above",
                "path": "Fixed straight box-centre XY target path plus centre yaw correction after automatic lift; physical landing and native success verification follow.",
                "state_access": "Uses native box-centre pose targets; generated controller data, not human demonstrations.",
            })
        writer = LeRobotWriter(output, recording_schema(environment), task, provenance=provenance,
                               max_frames_per_file=1)
        print(f"Dataset: {output.resolve()}", flush=True)
        if args.straight_line:
            started = perf_counter()
            for episode_index in range(args.episodes):
                episode_seed = args.seed + episode_index
                environment.reset(episode_seed)
                initial_offset = environment.environment.pose[0][:2].tolist()
                result = record_straight_episode(environment, writer, max_steps=max_steps,
                                                 speed_m_s=args.speed_mm_s / 1000,
                                                 yaw_speed_rad_s=math.radians(args.yaw_speed_deg_s))
                generated.append({"episode_index": episode_index, "seed": episode_seed,
                                  "initial_offset_metres": initial_offset, **result})
                report = write_collection_report(output, generated, args.episodes, perf_counter() - started)
                print(json.dumps({"completed": len(generated), "requested": args.episodes,
                                  "successful": report["successful_episodes"],
                                  "seed": episode_seed, "frames": result["frames"],
                                  "initial_xy_mm": [value * 1000 for value in initial_offset],
                                  "initial_yaw_deg": result["initial_yaw_deg"],
                                  "xy_error_mm": result["final_xy_error_m"] * 1000,
                                  "yaw_error_deg": math.degrees(result["final_yaw_error_rad"]),
                                  "max_cross_track_mm": result["max_cross_track_error_m"] * 1000,
                                  "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
        elif actions is not None:
            environment.reset(args.seed)
            result = record_episode(environment, actions, writer, max_steps=max_steps)
            replay_saved = True
            print(json.dumps(result), flush=True)
        else:
            session = DemonstrationSession(environment, writer, args.seed, max_steps)
            print(("Space starts/pauses; arrows control XY; Q/E rotate yaw about box centre; Enter lands then saves; Esc lands, saves and quits."
                   if args.env == "pushbox3d" else
                   "Space starts/pauses; Enter saves; R discards; Esc saves partial and quits."), flush=True)
            try:
                run_window(session, args.episodes)
            except KeyboardInterrupt:
                if writer.aborted:
                    raise
                session.finish_and_save("keyboard_interrupt")
    finally:
        try:
            if session is not None:
                session.close()
            if writer is not None and (generated or replay_saved or (session is not None and session.saved)):
                writer.close()
        finally:
            environment.close()
    if generated or replay_saved or (session is not None and session.saved):
        print(json.dumps(validate_dataset(output), indent=2))
    else:
        print("No demonstrations saved; no dataset was finalized.")


if __name__ == "__main__":
    main()
