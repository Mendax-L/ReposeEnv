"""Six stable stage IDs with sentence-level grasp instructions."""
from __future__ import annotations

import numpy as np


STAGE_LABELS = ("接近", "加紧", "抬起", "搬运对齐", "放下", "松开撤离")
STAGE_INSTRUCTIONS = (
    (
        "请让双臂夹爪接近上箱两侧的抓取位置，并调整好抓取姿态。",
        "请将两侧夹爪移动到上箱的抓取位置，做好夹持准备。",
        "请控制双臂靠近上箱两侧，调整夹爪位置和朝向以准备抓取。",
        "请把双臂夹爪对准上箱两侧的侧壁，并平稳接近抓取位置。",
    ),
    (
        "请逐渐闭合两侧夹爪，稳稳夹住上箱两侧的侧壁。",
        "请让双侧夹爪缓慢收紧，保持上箱稳定并建立可靠夹持。",
        "请平稳合拢两侧夹爪，将上箱夹紧以准备抬起。",
        "请控制两个夹爪逐步夹紧上箱，避免夹持时晃动箱体。",
    ),
    (
        "请保持双臂协同，平稳抬起已经夹紧的上箱。",
        "请在夹爪夹持稳定后，让双臂一起将上箱向上抬起。",
        "请稳稳夹住上箱，并用双臂同步将它抬离下箱。",
        "请保持两侧夹爪的夹持，让上箱缓慢上升并离开支撑面。",
    ),
    (
        "请保持上箱稳定，调整它的位置和朝向，使它与下箱对齐。",
        "请协同移动双臂并调整上箱朝向，将上箱对准下箱。",
        "请在夹持上箱的同时调整箱体位置和角度，使上下箱对齐。",
        "请平稳搬运上箱并修正它的朝向，让它对齐下箱的放置位置。",
    ),
    (
        "请保持上下箱对齐，缓慢降低上箱，直到它平稳落在下箱上。",
        "请让双臂同步下降，将对齐后的上箱轻轻放到下箱上。",
        "请稳定夹持上箱并缓慢下放，保持它与下箱对齐直至落稳。",
        "请在保持箱体对齐的同时降低双臂，让上箱平稳落在下箱上。",
    ),
    (
        "请松开两侧夹爪并让双臂撤离，等待上箱在下箱上保持稳定。",
        "请在上箱落稳后打开两个夹爪，让双臂离开箱体并等待箱体稳定。",
        "请逐步解除对上箱的夹持并撤回双臂，等待上下箱稳定叠放。",
        "请打开双侧夹爪并移开双臂，让上箱保持在下箱上稳定放置。",
    ),
)
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


def sample_stage_instructions(seed):
    """Choose one phrase per stage and episode without consuming physics RNG."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 73]))
    return tuple(variants[int(rng.integers(len(variants)))] for variants in STAGE_INSTRUCTIONS)


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
        "version": "grasp_stage_language/v2",
        "labels": [{"id": index, "text": text, "instructions": list(STAGE_INSTRUCTIONS[index])}
                   for index, text in enumerate(STAGE_LABELS)],
        "expert_stages": list(EXPERT_STAGES),
        "expert_stage_label_ids": list(EXPERT_STAGE_LABEL_IDS),
        "alignment": "Each label describes the same pre-action row and its end-effector action.",
        "task_index": "Per-frame instruction index into meta/tasks.parquet, not the stage ID; each episode uses one sentence per stage.",
        "sampling": "One instruction per stage sampled from the episode seed using an independent RNG stream; fixed throughout that stage, including its internal substages.",
        "policy_condition": "language.stage is a 6-D one-hot stage condition; language.stage_id is its integer ID.",
        "unlabelled": "stage_id=-1 and a zero vector when no expert stage has been selected.",
    }
