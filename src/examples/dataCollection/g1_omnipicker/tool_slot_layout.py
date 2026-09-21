"""Five-tool rack layouts: swap shelf-Y slots, keep each tool's x/z/pose.

Slot order 0..4 is left-to-right on the rack. assignment[slot] = tool index.
"""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

TOOL_NAMES = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]

TOOL_BODY_JOINT_NAMES = [
    "Group_Interactive_Spanner_task_spanner_joint",
    "Group_Interactive_Screwdriver_task_screwdriver_joint",
    "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_joint",
    "Group_Interactive_Flashlight_task_flashlight_joint",
    "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_joint",
]

# Robot-base frame. Randomization only replaces slot Y.
TOOL_REFERENCE_POS_B = np.asarray(
    [
        [0.5654831, -0.0693993, 0.1514528],
        [0.5599098, -0.1873288, 0.1568153],
        [0.5764828, -0.3049245, 0.1447767],
        [0.5079708, -0.4143662, 0.1536523],
        [0.5802917, -0.5180783, 0.1439366],
    ],
    dtype=np.float64,
)
TOOL_REFERENCE_QUAT_XYZW_B = np.asarray(
    [
        [-0.0, 0.7071068, 0.7071067, 0.0],
        [0.4999249, -0.5000751, -0.4999244, 0.5000756],
        [-0.0, 0.7071065, 0.7071071, 0.0],
        [-0.5, 0.5, -0.4999995, 0.5000005],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# Fixed lexicographic order of all 5! assignments.
LAYOUT_PERMUTATIONS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(5)))
N_LAYOUTS = len(LAYOUT_PERMUTATIONS)


def assignment_at(index: int) -> np.ndarray:
    """Return assignment[slot]=tool for a saved-episode index (0-based)."""
    if index < 0 or index >= N_LAYOUTS:
        raise IndexError(f"layout index {index} out of range 0..{N_LAYOUTS - 1}")
    return np.asarray(LAYOUT_PERMUTATIONS[index], dtype=np.int64)


def tools_left_to_right(assignment) -> list[str]:
    assignment = np.asarray(assignment, dtype=np.int64)
    return [TOOL_NAMES[int(tool_idx)] for tool_idx in assignment]


def layout_record(index: int, count: int) -> dict[str, Any]:
    assignment = assignment_at(index)
    names = tools_left_to_right(assignment)
    return {
        "tool_layout_index": int(index),
        "tool_layout_count": int(count),
        "assignment": assignment.tolist(),
        "tools_left_to_right": names,
        "layout_label": " ".join(names),
    }


def place_tools(env, base_body: str, assignment) -> None:
    """Set the five tool free joints. assignment[slot_idx] = tool_idx."""
    assignment = np.asarray(assignment, dtype=np.int64)
    if assignment.shape != (5,) or sorted(assignment.tolist()) != list(range(5)):
        raise ValueError(f"assignment 必须是 0..4 的排列，收到: {assignment.tolist()}")

    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_quat_wxyz = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
    base_rot = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])

    target_qpos = {}
    for slot_idx, tool_idx in enumerate(assignment):
        target_pos_b = TOOL_REFERENCE_POS_B[tool_idx].copy()
        target_pos_b[1] = TOOL_REFERENCE_POS_B[slot_idx, 1]
        target_rot_b = R.from_quat(TOOL_REFERENCE_QUAT_XYZW_B[tool_idx])
        world_pos = base_pos + base_rot.apply(target_pos_b)
        world_quat_xyzw = (base_rot * target_rot_b).as_quat()
        world_quat_wxyz = world_quat_xyzw[[3, 0, 1, 2]]
        target_qpos[TOOL_BODY_JOINT_NAMES[tool_idx]] = np.concatenate(
            [world_pos, world_quat_wxyz]
        )

    env.set_joint_qpos(target_qpos)
    env.mj_forward()
