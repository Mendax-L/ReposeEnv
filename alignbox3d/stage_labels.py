"""Six shared language labels for the end-effector grasp demonstration."""
from __future__ import annotations

import numpy as np


STAGE_LABELS = ("接近", "加紧", "抬起", "搬运对齐", "放下", "松开撤离")
EXPERT_STAGES = (
    "initial_settle", "approach_above", "approach_rim", "orient_grasp",
    "descend_to_grasp", "close_grippers", "lift", "carry_align", "lower",
    "release", "withdraw_rim", "withdraw_orient", "withdraw_above", "park",
    "verify_released_alignment",
)
EXPERT_STAGE_LABEL_IDS = (0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5)
LANGUAGE_FEATURES = {
    "language.stage_id": {"dtype": "int64", "shape": [1], "names": ["stage_id"]},
    "language.stage": {"dtype": "float32", "shape": [6], "names": list(STAGE_LABELS)},
}


def stage_observation(stage_id):
    """Unlabelled recordings use -1 and a zero vector, never a fabricated label."""
    if type(stage_id) is not int or not -1 <= stage_id < len(STAGE_LABELS):
        raise ValueError("stage_id must be -1 or one of 0..5")
    one_hot = np.zeros(len(STAGE_LABELS), dtype=np.float32)
    if stage_id >= 0:
        one_hot[stage_id] = 1.
    return {"language.stage_id": np.asarray([stage_id], dtype=np.int64),
            "language.stage": one_hot}


def language_metadata():
    return {
        "version": "grasp_stage_language/v1",
        "labels": [{"id": index, "text": text} for index, text in enumerate(STAGE_LABELS)],
        "expert_stages": list(EXPERT_STAGES),
        "expert_stage_label_ids": list(EXPERT_STAGE_LABEL_IDS),
        "alignment": "Each label describes the same pre-action row and its end-effector action.",
        "task_index": "Per-frame label index into meta/tasks.parquet; an episode contains six stage tasks.",
        "policy_condition": "language.stage is a 6-D one-hot stage condition; language.stage_id is its integer ID.",
        "unlabelled": "stage_id=-1 and a zero vector when no expert stage has been selected.",
    }
