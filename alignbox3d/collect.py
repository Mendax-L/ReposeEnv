"""Record randomized, contact-only grasp/carry/release demonstrations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from common.lerobot import LeRobotWriter, validate_dataset
from common.files import write_json
from common.recording import recording_schema
from alignbox3d.diversity import DiverseAlignBox3DEnv, VISUAL_MODES
from alignbox3d.grasp_collection import GraspConfig, EXPERT_STAGES
from alignbox3d.parallel_collection import collection_attempts
from alignbox3d.collection_metadata import metadata_dictionary
from alignbox3d.stage_labels import STAGE_LABELS, language_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="Number of successful, fully released demonstrations.")
    parser.add_argument("--max-attempts", type=int, help="Default: five attempts per requested success.")
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--visual-mode", choices=VISUAL_MODES, default="all")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=1, help="Independent simulation processes; successful episodes merge in seed order.")
    parser.add_argument("--approach-speed-mm-s", type=float, default=50.)
    parser.add_argument("--withdraw-speed-mm-s", type=float, default=50.)
    parser.add_argument("--free-angular-speed-deg-s", type=float, default=24.)
    parser.add_argument("--contact-speed-mm-s", type=float, default=15.)
    parser.add_argument("--lift-speed-mm-s", type=float, default=20.)
    parser.add_argument("--carry-speed-mm-s", type=float, default=20.)
    parser.add_argument("--lower-speed-mm-s", type=float, default=20.)
    parser.add_argument("--angular-speed-deg-s", type=float, default=12.)
    parser.add_argument("--acceleration-mm-s2", type=float, help="Optional motion acceleration limit; omitted keeps the original motion timing.")
    parser.add_argument("--gripper-speed-mm-s", type=float, default=4.)
    parser.add_argument("--grasp-y-mm", type=float, default=10.)
    parser.add_argument("--grasp-z-mm", type=float, default=5.)
    parser.add_argument("--pair-y-difference-mm", type=float, default=2.)
    parser.add_argument("--pair-z-difference-mm", type=float, default=5.)
    parser.add_argument("--angle-jitter-deg", type=float, default=1.5)
    parser.add_argument("--independent-angle-deg", type=float, default=.25)
    parser.add_argument("--initial-tilt-deg", type=float, default=2.)
    parser.add_argument("--lift-mm", type=float, default=20.)
    args = parser.parse_args()
    args.max_attempts = args.max_attempts if args.max_attempts is not None else 5*args.episodes
    if (min(args.episodes, args.max_attempts, args.max_steps, args.workers) < 1
            or args.max_attempts < args.episodes
            or not 0 <= args.seed < 2**63-args.max_attempts):
        parser.error("Counts must be positive, max-attempts >= episodes, and seeds must fit int64.")
    if args.render_size < 16 or args.render_size % 2:
        parser.error("render-size must be even and at least 16.")
    if not np.isfinite(args.initial_tilt_deg) or not 0 <= args.initial_tilt_deg <= 5:
        parser.error("initial-tilt-deg must be within [0,5].")
    try:
        config = GraspConfig(point_y_range_m=args.grasp_y_mm/1000,
            point_z_range_m=args.grasp_z_mm/1000, pair_y_difference_m=args.pair_y_difference_mm/1000,
            pair_z_difference_m=args.pair_z_difference_mm/1000, angle_jitter_degrees=args.angle_jitter_deg,
            independent_angle_degrees=args.independent_angle_deg, lift_m=args.lift_mm/1000,
            approach_speed_m_s=args.approach_speed_mm_s/1000, withdraw_speed_m_s=args.withdraw_speed_mm_s/1000,
            free_angular_speed_degrees_s=args.free_angular_speed_deg_s,
            contact_speed_m_s=args.contact_speed_mm_s/1000,
            lift_speed_m_s=args.lift_speed_mm_s/1000, carry_speed_m_s=args.carry_speed_mm_s/1000,
            lower_speed_m_s=args.lower_speed_mm_s/1000, angular_speed_degrees_s=args.angular_speed_deg_s,
            limit_motion_acceleration=args.acceleration_mm_s2 is not None,
            linear_acceleration_m_s2=(args.acceleration_mm_s2/1000 if args.acceleration_mm_s2 is not None else .030), gripper_speed_m_s=args.gripper_speed_mm_s/1000)
    except ValueError as error:
        parser.error(str(error))
    base = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or base/"training_data/lerobot/alignbox3d"/f"grasp-carry-{stamp}"
    if output.exists():
        parser.error(f"Output exists; choose a new directory: {output}")
    environment = DiverseAlignBox3DEnv(render_size=args.render_size)
    writer = None
    records = []
    saved = 0
    try:
        environment.reset(args.seed, options={"terminate_on_alignment": False})
        sources = ("alignbox3d/grasp_collection.py", "alignbox3d/parallel_collection.py", "alignbox3d/geometry.py", "alignbox3d/env.py",
                   "alignbox3d/native/env.py", "alignbox3d/native/model.py", "alignbox3d/native/kinematics.py",
                   "alignbox3d/diversity.py", "common/schema.py", "common/recording.py", "common/lerobot.py",
                   "common/files.py", "alignbox3d/collection_metadata.py", "alignbox3d/stage_labels.py",
                   "alignbox3d/collect.py")
        snapshots = {name: (base/name).read_bytes() for name in sources}
        schema = recording_schema(environment)
        schema["grasp_expert"] = {"config": asdict(config), "stages": list(EXPERT_STAGES),
            "initial_tilt_range_degrees": [-args.initial_tilt_deg, args.initial_tilt_deg],
            "sampling_frame": "upper-box local frame, origin at bottom centre",
            "box_path_frame": "shared world; path follows the geometric box centre",
            "acceptance": "physical lift, aligned placement, open grippers, withdrawal and stable released alignment",
            "selection_bias": "only physically successful and reachable attempts are saved; all rejected attempts remain in collection.json"}
        provenance = {"kind": "scripted_contact_grasp_carry_release", "human_demonstration": False,
            "arguments": {**vars(args), "output": str(output)}, "environment_contract": environment.contract(),
            "source_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in snapshots.items()},
            "source_snapshot": "meta/source_snapshot",
            "gripper_servo": {"kp_N_per_m": 2500, "kv_N_s_per_m": 40, "force_limit_N": 10},
            "runtime": {"python": sys.version, "mujoco": __import__("mujoco").__version__}}
        writer = LeRobotWriter(output, schema, STAGE_LABELS[0],
            provenance=provenance, max_frames_per_file=1)
        for name, value in snapshots.items():
            path = output/"meta/source_snapshot"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        write_json(output/"meta/diversity.json", metadata_dictionary(environment.visual_config))
        write_json(output/"meta/language.json", language_metadata())
        print(f"Dataset: {output.resolve()}", flush=True)
        with collection_attempts(environment, writer, args, config) as attempts:
            for result in attempts:
                worker_dataset = result.pop("worker_dataset", None)
                if result["saved"] and worker_dataset is not None:
                    writer.import_episode(worker_dataset)
                result["episode_index"] = saved if result["saved"] else None
                saved += bool(result["saved"])
                records.append(result)
                write_json(output/"meta/collection.json", {"kind": provenance["kind"],
                    "requested_episodes": args.episodes, "saved_episodes": saved,
                    "attempts": len(records), "failed_attempts": len(records)-saved,
                    "workers": args.workers, "ordering": "seed_ascending",
                    "complete": saved == args.episodes, "episodes": records})
                print(json.dumps({key: result.get(key) for key in
                    ("seed", "worker_pid", "saved", "frames", "failure_reason", "peak_lift_m")}), flush=True)
        if not saved:
            writer.abort()
            raise RuntimeError("No successful demonstrations; inspect meta/collection.json.")
        writer.close()
        print(json.dumps(validate_dataset(output)), flush=True)
        if saved < args.episodes:
            raise RuntimeError(f"Saved {saved}/{args.episodes} demonstrations within max-attempts; inspect collection.json.")
    finally:
        if writer is not None and not writer.aborted:
            if saved:
                writer.close()
            else:
                writer.abort()
        environment.close()


if __name__ == "__main__":
    main()
