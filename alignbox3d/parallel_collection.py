"""Isolated spawn workers with deterministic seeds and a single dataset merger."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from common.lerobot import LeRobotWriter
from .diversity import DiverseAlignBox3DEnv, is_heldout_configuration
from .grasp_collection import GraspConfig, collect_grasp_episode
from .stage_labels import STAGE_LABELS


_worker_environment = None


def initialize_worker(render_size):
    global _worker_environment
    _worker_environment = DiverseAlignBox3DEnv(render_size=render_size)


def run_attempt(environment, writer, seed, attempt, settings, config):
    started = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 72]))
    tilt = np.deg2rad(rng.uniform(-settings["initial_tilt_deg"], settings["initial_tilt_deg"], 2))
    environment.reset(seed, options={"terminate_on_alignment": False,
        "roll": float(tilt[0]), "pitch": float(tilt[1]), "visual_mode": settings["visual_mode"]})
    initial = environment.environment.data.qpos[
        environment.environment.box_qadr:environment.environment.box_qadr+7].copy().tolist()
    if is_heldout_configuration(environment.visual_metadata):
        result = {"seed": seed, "saved": False, "success": False,
                  "failure_reason": "heldout_visual_signature", "frames": 0}
    else:
        result = collect_grasp_episode(environment, writer, seed, config=config,
                                       max_steps=settings["max_steps"])
    result.update(attempt=attempt, initial_box_pose_world=initial,
                  initial_roll_pitch_rad=tilt.tolist(), visual_parameters=environment.episode_metadata(),
                  worker_pid=os.getpid(), attempt_seconds=perf_counter()-started)
    return result


def worker_attempt(seed, attempt, settings, config_values, schema, scratch):
    environment = _worker_environment
    root = Path(scratch)/f"attempt-{attempt:08d}"
    writer = LeRobotWriter(root, schema, STAGE_LABELS[0],
        provenance={"kind": "isolated_grasp_worker", "seed": seed, "pid": os.getpid()},
        max_frames_per_file=1)
    try:
        result = run_attempt(environment, writer, seed, attempt, settings, GraspConfig(**config_values))
        if result["saved"]:
            writer.close()
            result["worker_dataset"] = str(root)
        else:
            writer.abort()
        return result
    except BaseException:
        writer.abort()
        raise


@contextmanager
def collection_attempts(environment, writer, args, config):
    """Yield seed-ordered results; workers never share a renderer or output file.

    At most remaining-success-count jobs are in flight, so successful episodes
    are never discarded merely because a slower, earlier attempt finished late.
    """
    from dataclasses import asdict
    settings = {"initial_tilt_deg": args.initial_tilt_deg,
                "visual_mode": args.visual_mode, "max_steps": args.max_steps}
    if args.workers == 1:
        def serial():
            saved = 0
            for attempt in range(args.max_attempts):
                result = run_attempt(environment, writer, args.seed+attempt, attempt, settings, config)
                yield result
                saved += bool(result["saved"])
                if saved == args.episodes:
                    return
        yield serial()
        return
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    with TemporaryDirectory(prefix=".grasp-workers-", dir=writer.root) as scratch:
        if not Path(scratch).resolve().is_relative_to(writer.root.resolve()):
            raise RuntimeError("Worker scratch directory escaped the destination dataset.")
        executor = ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
            initargs=(args.render_size,))
        pending = {}
        def parallel():
            submitted = saved = 0
            for attempt in range(args.max_attempts):
                window = min(args.workers, args.episodes-saved)
                while submitted < args.max_attempts and len(pending) < window:
                    pending[submitted] = executor.submit(worker_attempt, args.seed+submitted,
                        submitted, settings, asdict(config), writer.schema, scratch)
                    submitted += 1
                if attempt not in pending:
                    return
                result = pending.pop(attempt).result()
                yield result
                saved += bool(result["saved"])
                if saved == args.episodes:
                    return
        try:
            yield parallel()
        finally:
            for future in pending.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
