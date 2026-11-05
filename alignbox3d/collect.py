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
from common.config import load_environment_config
from alignbox3d.diversity import DEFAULT_DATA_CONFIG, DiverseAlignBox3DEnv, VISUAL_MODES, load_data_config
from alignbox3d.grasp_collection import GraspConfig, EXPERT_STAGES
from alignbox3d.parallel_collection import collection_attempts
from alignbox3d.collection_metadata import metadata_dictionary
from alignbox3d.stage_labels import STAGE_INSTRUCTIONS, language_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="Number of successful, fully released demonstrations.")
    parser.add_argument("--max-attempts", type=int, help="Default: five attempts per requested success.")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG, help="Shared training/validation scene config (JSON syntax in a legacy .yml file).")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--episode-mode", choices=("mixed", "normal", "offset_recovery", "contact_error", "grasp_ready"),
                        default="mixed", help="Force one trajectory type per output directory; mixed uses the recovery fractions.")
    parser.add_argument("--seed", type=int, help="First attempt seed; defaults to the selected split's range start.")
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--visual-mode", choices=VISUAL_MODES, default="all")
    parser.add_argument("--max-steps", type=int, default=100 * load_environment_config("alignbox3d")["fps"])
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
    parser.add_argument("--initial-tool-jitter-mm", type=float, default=8.,
                        help="Independent XYZ jitter of each parked gripper, within a safe reachability window.")
    parser.add_argument("--recovery-fraction", type=float, default=.25,
                        help="Fraction of attempts recorded from a disturbed approach or held-box carry state.")
    parser.add_argument("--recovery-offset-mm", type=float, default=10.,
                        help="Maximum approach-tool or held-box displacement for recovery demonstrations.")
    parser.add_argument("--contact-error-fraction", type=float, default=0.,
                        help="Fraction of attempts starting from measured wrong box contact with both grippers laterally offset.")
    parser.add_argument("--grasp-ready-fraction", type=float, default=0.,
                        help="Fraction of attempts recorded from a verified rim-between-fingers pose, immediately before closing.")
    parser.add_argument("--contact-error-lateral-mm", type=float, default=19.,
                        help="Both grippers' box-local lateral offsets for failed contact, 18-20 mm.")
    parser.add_argument("--lift-mm", type=float, default=20.)
    args = parser.parse_args()
    try:
        data_config = load_data_config(args.data_config)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.data_config = args.data_config.resolve()
    seed_start, seed_stop = data_config["splits"][args.split]["seed_range"]
    args.seed = seed_start if args.seed is None else args.seed
    args.max_attempts = args.max_attempts if args.max_attempts is not None else 5*args.episodes
    if (min(args.episodes, args.max_attempts, args.max_steps, args.workers) < 1
            or args.max_attempts < args.episodes
            or not 0 <= args.seed < 2**63-args.max_attempts):
        parser.error("Counts must be positive, max-attempts >= episodes, and seeds must fit int64.")
    if not seed_start <= args.seed < args.seed+args.max_attempts <= seed_stop:
        parser.error(f"All attempt seeds must lie in the {args.split} seed_range [{seed_start}, {seed_stop}).")
    if args.split == "validation" and args.visual_mode != "all":
        parser.error("Validation uses the configured industrial combinations; visual-mode must be all.")
    if args.render_size < 16 or args.render_size % 2:
        parser.error("render-size must be even and at least 16.")
    if not np.isfinite(args.initial_tilt_deg) or not 0 <= args.initial_tilt_deg <= 5:
        parser.error("initial-tilt-deg must be within [0,5].")
    if (not np.isfinite([args.initial_tool_jitter_mm, args.recovery_fraction,
                         args.recovery_offset_mm, args.contact_error_fraction, args.grasp_ready_fraction,
                         args.contact_error_lateral_mm]).all()
            or not 0 <= args.initial_tool_jitter_mm <= 15
            or not 0 <= args.recovery_fraction <= 1
            or not 0 <= args.contact_error_fraction <= 1
            or not 0 <= args.grasp_ready_fraction <= 1
            or args.recovery_fraction+args.contact_error_fraction+args.grasp_ready_fraction > 1
            or not 0 <= args.recovery_offset_mm <= 15
            or not 18 <= args.contact_error_lateral_mm <= 20):
        parser.error("Jitter/offset must be 0-15 mm; contact error lateral offset 18-20 mm; recovery fractions must sum to at most 1.")
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
    output = args.output or base/"training_data/lerobot/alignbox3d"/f"grasp-carry-{args.split}-{stamp}"
    if output.exists():
        parser.error(f"Output exists; choose a new directory: {output}")
    environment = DiverseAlignBox3DEnv(render_size=args.render_size, split=args.split, data_config=args.data_config)
    writer = None
    records = []
    saved = 0
    try:
        environment.reset(args.seed, options={"terminate_on_alignment": False})
        sources = ("alignbox3d/grasp_collection.py", "alignbox3d/parallel_collection.py", "alignbox3d/geometry.py", "alignbox3d/env.py",
                   "alignbox3d/native/env.py", "alignbox3d/native/model.py", "alignbox3d/native/kinematics.py",
                   "alignbox3d/diversity.py", "common/schema.py", "common/recording.py", "common/lerobot.py",
                   "common/files.py", "alignbox3d/collection_metadata.py", "alignbox3d/stage_labels.py",
                   "alignbox3d/collect.py", "common/environment.py", "common/config.py", "config/alignbox3d.yml",
                   "config/alignbox3d_data.yml")
        snapshots = {name: (base/name).read_bytes() for name in sources}
        snapshots["config/alignbox3d_data.yml"] = args.data_config.read_bytes()
        schema = recording_schema(environment)
        schema["grasp_expert"] = {"config": asdict(config), "stages": list(EXPERT_STAGES),
            "episode_mode": args.episode_mode,
            "initial_tilt_range_degrees": [-args.initial_tilt_deg, args.initial_tilt_deg],
            "sampling_frame": "upper-box local frame, origin at bottom centre",
            "box_path_frame": "shared world; path follows the geometric box centre",
            "initial_tool_jitter_mm": args.initial_tool_jitter_mm,
            "recovery_fraction": args.recovery_fraction,
            "recovery_offset_mm": args.recovery_offset_mm,
            "contact_error_fraction": args.contact_error_fraction,
            "grasp_ready_fraction": args.grasp_ready_fraction,
            "contact_error_lateral_mm": args.contact_error_lateral_mm,
            "recovery_sampling": "Recovery episodes begin recording after an unrecorded approach/carry displacement or measured wrong box contact. Both grippers are offset outside/inside their walls in four deterministic combinations; finger-pad geometry is checked before saving. Contact-error episodes record withdrawal, replanning, regrasp and final success; mistake-making actions are excluded. Grasp-ready episodes begin with both walls between fingers and immediately close.",
            "acceptance": "physical lift, aligned placement, open grippers, withdrawal and stable released alignment",
            "selection_bias": "only physically successful and reachable attempts are saved; all rejected attempts remain in collection.json"}
        provenance = {"kind": "scripted_contact_grasp_carry_release", "human_demonstration": False,
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in {**vars(args), "output": output}.items()},
            "data_split": schema["data_split"], "environment_contract": environment.contract(),
            "source_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in snapshots.items()},
            "source_snapshot": "meta/source_snapshot",
            "gripper_servo": {"kp_N_per_m": 2500, "kv_N_s_per_m": 40, "force_limit_N": 10},
            "runtime": {"python": sys.version, "mujoco": __import__("mujoco").__version__}}
        writer = LeRobotWriter(output, schema, STAGE_INSTRUCTIONS[0][0],
            provenance=provenance, max_frames_per_file=1)
        for name, value in snapshots.items():
            path = output/"meta/source_snapshot"/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        write_json(output/"meta/diversity.json", metadata_dictionary(environment.visual_config, environment.data_config))
        write_json(output/"meta/data_split.json", schema["data_split"])
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
                    "saved_by_recovery_stage": {name: sum(bool(row["saved"]) and
                        (row.get("recovery_stage") or "none") == name for row in records)
                        for name in ("none", "approach", "carry", "contact_error", "grasp_ready")},
                    "saved_by_contact_pattern": {"/".join(pattern): sum(bool(row["saved"]) and
                        (row.get("recovery_start") or {}).get("pattern") == list(pattern) for row in records)
                        for pattern in (("outside", "outside"), ("inside", "inside"),
                                        ("outside", "inside"), ("inside", "outside"))},
                    "workers": args.workers, "split": args.split, "ordering": "seed_ascending",
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
