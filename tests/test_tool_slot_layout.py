"""Tool rack full-permutation layouts; no MuJoCo / ORCA required."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "examples" / "dataCollection" / "g1_omnipicker"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_slot_layout import (  # noqa: E402
    LAYOUT_PERMUTATIONS,
    N_LAYOUTS,
    TOOL_NAMES,
    assignment_at,
    layout_record,
    tools_left_to_right,
)
from dataStorage.lerobot_classified import (  # noqa: E402
    DOUBTFUL,
    GOOD,
    QUALIFIED,
    ClassifiedLeRobotHub,
    CATEGORY_DIR,
)


class ToolSlotLayoutTests(unittest.TestCase):
    def test_all_120_permutations_unique(self):
        self.assertEqual(N_LAYOUTS, 120)
        self.assertEqual(len(LAYOUT_PERMUTATIONS), 120)
        as_tuples = [tuple(p) for p in LAYOUT_PERMUTATIONS]
        self.assertEqual(len(set(as_tuples)), 120)
        for perm in as_tuples:
            self.assertEqual(sorted(perm), [0, 1, 2, 3, 4])

    def test_resume_index_matches_saved_count(self):
        first = assignment_at(0)
        self.assertEqual(first.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(tools_left_to_right(first), TOOL_NAMES)
        rec = layout_record(37, 120)
        self.assertEqual(rec["tool_layout_index"], 37)
        self.assertEqual(rec["tool_layout_count"], 120)
        self.assertEqual(rec["assignment"], list(LAYOUT_PERMUTATIONS[37]))
        self.assertEqual(len(rec["tools_left_to_right"]), 5)
        with self.assertRaises(IndexError):
            assignment_at(120)

    def test_hub_total_episodes_reads_disk_for_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = ClassifiedLeRobotHub.__new__(ClassifiedLeRobotHub)
            hub.parent_root = tmp
            hub._writers = {}
            for cat, n in ((GOOD, 10), (QUALIFIED, 20), (DOUBTFUL, 7)):
                meta = Path(tmp) / CATEGORY_DIR[cat] / "meta"
                meta.mkdir(parents=True)
                (meta / "info.json").write_text(json.dumps({"total_episodes": n}))
            self.assertEqual(hub.total_episodes, 37)
            rec = layout_record(hub.total_episodes, 120)
            self.assertEqual(rec["assignment"], list(LAYOUT_PERMUTATIONS[37]))


if __name__ == "__main__":
    unittest.main()
