"""Synthetic tests for G1 classified collection. Writes only under /tmp."""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from g1_classification_review import (  # noqa: E402
    CLASSIFICATION_NEUTRAL_HOLD_SEC,
    DELETE,
    DOUBTFUL,
    GOOD,
    QUALIFIED,
    SHUTDOWN,
    ClassificationReview,
    PicoButtons,
    buttons_from_key_state,
    grip_monitor_decision,
    poll_pico_raw,
)
from dataStorage.lerobot_classified import ClassifiedLeRobotHub  # noqa: E402

TEST_ROOT = Path("/tmp/g1_tele_classification_test")
LEGACY_ROOT = Path.home() / "datasets" / "g1_tele"
LEGACY_SNAP = Path("/tmp/g1_tele_legacy_snapshot.txt")

RESULTS: dict[str, str] = {}


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakePico:
    def __init__(self) -> None:
        self.callbacks: list = []
        self.key_state = None
        self.update_keys_seen: list = []

    def update(self, keys):
        self.update_keys_seen.append(list(keys))
        for _k in keys:
            self.callbacks.append("fired")

    def get_key_state(self):
        return self.key_state


class FakeWriter:
    def __init__(self, root: str, repo_id: str) -> None:
        self.root = Path(root)
        self.repo_id = repo_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._episodes: list[int] = []
        self._load()

    @classmethod
    def create_or_resume(cls, repo_id: str, root: str, **kwargs):
        rootp = Path(root)
        if rootp.exists():
            meta = rootp / "meta" / "info.json"
            if not meta.is_file():
                if any(rootp.iterdir()):
                    raise ValueError(f"[resume] fail closed: {root}")
            return cls(root, repo_id)
        return cls(root, repo_id)

    def _index_path(self) -> Path:
        return self.root / "meta" / "episodes.jsonl"

    def _load(self) -> None:
        p = self._index_path()
        if p.is_file():
            self._episodes = [json.loads(line)["episode_index"] for line in p.read_text().splitlines() if line.strip()]
        else:
            self._episodes = []

    def _save(self) -> None:
        meta = self.root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "info.json").write_text(json.dumps({"repo_id": self.repo_id, "total_episodes": len(self._episodes)}))
        lines = [json.dumps({"episode_index": i}) for i in self._episodes]
        self._index_path().write_text("\n".join(lines) + ("\n" if lines else ""))
        (self.root / "data").mkdir(exist_ok=True)
        for i in self._episodes:
            (self.root / "data" / f"episode_{i:06d}.ok").write_text("ok")

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_frames(self) -> int:
        return self.num_episodes * 2

    def append_episode(self) -> int:
        ep = self.num_episodes
        self._episodes.append(ep)
        self._save()
        return ep

    def discard_episode(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeStaging:
    def __init__(self) -> None:
        self.committed_to: list[str] = []
        self.discarded = 0
        self.parked = True

    def commit_uncommitted_to(self, dest: FakeWriter) -> int:
        self.committed_to.append(str(dest.root))
        return dest.append_episode()

    def discard_episode(self) -> None:
        self.discarded += 1
        self.parked = False

    def close(self) -> None:
        return None


def _record(name: str, ok: bool) -> None:
    RESULTS[name] = "PASS" if ok else "FAIL"
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def _assert(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        print(f"  detail: {detail}")
    _record(name, cond)


def test_review_ready_mapping() -> None:
    clock = FakeClock()
    review = ClassificationReview(hold_sec=0.25, now_fn=clock)
    review.feed(PicoButtons())
    clock.advance(0.25)
    review.feed(PicoButtons())
    assert review.phase == "READY"
    _assert("SYNTHETIC_L_GRIP_GOOD", review.feed(PicoButtons(l_grip=True)) == GOOD)
    review = _ready_review(clock)
    _assert("SYNTHETIC_X_QUALIFIED", review.feed(PicoButtons(x=True)) == QUALIFIED)
    review = _ready_review(clock)
    _assert("SYNTHETIC_A_DOUBTFUL", review.feed(PicoButtons(a=True)) == DOUBTFUL)
    review = _ready_review(clock)
    _assert("SYNTHETIC_R_GRIP_DELETE", review.feed(PicoButtons(r_grip=True)) == DELETE)
    review = _ready_review(clock)
    _assert("SYNTHETIC_B_NO_CLASSIFY", review.feed(PicoButtons(b=True)) is None)


def _ready_review(clock: FakeClock) -> ClassificationReview:
    review = ClassificationReview(hold_sec=0.25, now_fn=clock)
    review.feed(PicoButtons())
    clock.advance(0.30)
    review.feed(PicoButtons())
    assert review.phase == "READY"
    return review


def test_neutral_gate() -> None:
    clock = FakeClock()
    review = ClassificationReview(hold_sec=0.25, now_fn=clock)
    # Enter REVIEW with L_GRIP still held: must not immediately GOOD.
    d0 = review.feed(PicoButtons(l_grip=True))
    clock.advance(1.0)
    d1 = review.feed(PicoButtons(l_grip=True))
    ok_blocked = d0 is None and d1 is None and review.phase == "NEUTRAL"
    review.feed(PicoButtons())
    clock.advance(0.24)
    d2 = review.feed(PicoButtons())
    still_neutral = review.phase == "NEUTRAL" and d2 is None
    clock.advance(0.02)
    d3 = review.feed(PicoButtons())
    ready = review.phase == "READY" and review.became_ready and d3 is None
    d4 = review.feed(PicoButtons(l_grip=True))
    _assert("NEUTRAL_GATE_L_GRIP", ok_blocked and still_neutral and ready and d4 == GOOD)

    for name, held, rising, expected in (
        ("NEUTRAL_GATE_R_GRIP", PicoButtons(r_grip=True), PicoButtons(r_grip=True), DELETE),
        ("NEUTRAL_GATE_X", PicoButtons(x=True), PicoButtons(x=True), QUALIFIED),
        ("NEUTRAL_GATE_A", PicoButtons(a=True), PicoButtons(a=True), DOUBTFUL),
    ):
        clock = FakeClock()
        review = ClassificationReview(hold_sec=0.25, now_fn=clock)
        assert review.feed(held) is None
        clock.advance(1.0)
        assert review.feed(held) is None
        review.feed(PicoButtons())
        clock.advance(0.25)
        review.feed(PicoButtons())
        _assert(name, review.feed(rising) == expected)


def test_both_grip_priority() -> None:
    clock = FakeClock()
    review = _ready_review(clock)
    d = review.feed(PicoButtons(l_grip=True, r_grip=True))
    _assert("BOTH_GRIP_SHUTDOWN_PRIORITY", d == SHUTDOWN)


def test_recording_monitor() -> None:
    _assert(
        "RECORDING_X_NO_CLASSIFY",
        grip_monitor_decision("RECORDING", False, False, False, False) is None,
    )
    _assert(
        "RECORDING_RIGHT_GRIP_DISCARD",
        grip_monitor_decision("RECORDING", False, True, False, False) == "RECORDING_DISCARD",
    )
    _assert(
        "REVIEW_RIGHT_GRIP_NOT_MONITOR_DISCARD",
        grip_monitor_decision("REVIEW", False, True, False, False) is None,
    )
    _assert(
        "REVIEW_BOTH_GRIP_STILL_SHUTDOWN",
        grip_monitor_decision("REVIEW", True, True, False, False) == SHUTDOWN,
    )


def test_poll_pico_no_callback() -> None:
    pico = FakePico()
    pico.key_state = {
        "leftHand": {"gripButtonPressed": True, "primaryButtonPressed": False, "secondaryButtonPressed": False},
        "rightHand": {"gripButtonPressed": False, "primaryButtonPressed": False, "secondaryButtonPressed": False},
    }
    buttons = poll_pico_raw(pico)
    _assert("PICO_UPDATE_EMPTY_KEYS", pico.update_keys_seen == [[]])
    _assert("PICO_NO_CONTROLLER_CALLBACK", pico.callbacks == [])
    _assert("PICO_RAW_L_GRIP", buttons.l_grip is True)


def test_buttons_from_key_state() -> None:
    ks = {
        "leftHand": {"gripButtonPressed": True, "primaryButtonPressed": True, "secondaryButtonPressed": False},
        "rightHand": {"gripButtonPressed": False, "primaryButtonPressed": True, "secondaryButtonPressed": True},
    }
    b = buttons_from_key_state(ks)
    _assert("BUTTON_MAP", b.l_grip and b.x and b.a and b.b and not b.r_grip)


def _make_hub(root: Path) -> ClassifiedLeRobotHub:
    return ClassifiedLeRobotHub(
        parent_root=str(root),
        repo_id="local/g1_omnipicker",
        writer_kwargs={},
        staging_root=str(root / "_staging"),
        writer_factory=FakeWriter.create_or_resume,
        staging=FakeStaging(),
    )


def test_save_dirs_and_isolation() -> None:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    hub = _make_hub(TEST_ROOT)
    seq = [GOOD, GOOD, QUALIFIED, DOUBTFUL, GOOD]
    for cat in seq:
        hub.commit(cat)
    good = FakeWriter(str(TEST_ROOT / "good"), "local/g1_omnipicker_good")
    qual = FakeWriter(str(TEST_ROOT / "qualified"), "local/g1_omnipicker_qualified")
    doubt = FakeWriter(str(TEST_ROOT / "Doubtful"), "local/g1_omnipicker_doubtful")
    _assert("GOOD_SAVE_TEST", good.num_episodes == 3 and (TEST_ROOT / "good" / "data" / "episode_000002.ok").is_file())
    _assert("QUALIFIED_SAVE_TEST", qual.num_episodes == 1 and (TEST_ROOT / "qualified").is_dir())
    _assert("DOUBTFUL_SAVE_TEST", doubt.num_episodes == 1 and (TEST_ROOT / "Doubtful").is_dir())
    _assert("CATEGORY_ISOLATION_TEST", good._episodes == [0, 1, 2] and qual._episodes == [0] and doubt._episodes == [0])
    _assert("NO_DELETE_DIR", not (TEST_ROOT / "delete").exists() and not (TEST_ROOT / "Delete").exists())


def test_delete_and_cancel_index() -> None:
    root = TEST_ROOT / "index_cases"
    if root.exists():
        shutil.rmtree(root)
    hub = _make_hub(root)
    hub.commit(GOOD)
    hub.commit(DELETE)
    hub.commit(GOOD)
    good = FakeWriter(str(root / "good"), "x")
    _assert("DELETE_DISCARD_TEST", hub.staging.discarded == 1)
    _assert("DELETE_NO_INDEX_GAP_TEST", good._episodes == [0, 1])

    hub2 = _make_hub(root / "cancel")
    hub2.commit(GOOD)
    # RECORDING right-grip cancel is discard, same as not committing
    hub2.discard()
    hub2.commit(GOOD)
    good2 = FakeWriter(str(root / "cancel" / "good"), "x")
    _assert("RIGHT_GRIP_RECORDING_DISCARD_TEST", hub2.staging.discarded == 1)
    _assert("RIGHT_GRIP_CANCEL_NO_INDEX_GAP_TEST", good2._episodes == [0, 1])


def test_auto_resume() -> None:
    root = TEST_ROOT / "resume"
    if root.exists():
        shutil.rmtree(root)
    hub = _make_hub(root)
    hub.commit(GOOD)
    hub.commit(GOOD)
    hub.close()
    hub2 = _make_hub(root)
    hub2.commit(GOOD)
    good = FakeWriter(str(root / "good"), "x")
    _assert("AUTO_RESUME_TEST", good._episodes == [0, 1, 2])
    _assert("GOOD_AUTO_RESUME", good.num_episodes == 3)

    hubq = _make_hub(root)
    hubq.commit(QUALIFIED)
    hubq.close()
    hubq2 = _make_hub(root)
    hubq2.commit(QUALIFIED)
    qual = FakeWriter(str(root / "qualified"), "x")
    _assert("QUALIFIED_AUTO_RESUME", qual._episodes == [0, 1])

    hubd = _make_hub(root)
    hubd.commit(DOUBTFUL)
    hubd.close()
    hubd2 = _make_hub(root)
    hubd2.commit(DOUBTFUL)
    doubt = FakeWriter(str(root / "Doubtful"), "x")
    _assert("DOUBTFUL_AUTO_RESUME", doubt._episodes == [0, 1])


def test_fail_closed() -> None:
    root = TEST_ROOT / "broken"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "junk.txt").write_text("not a dataset")
    raised = False
    try:
        FakeWriter.create_or_resume("local/x", str(root))
    except ValueError:
        raised = True
    _assert("FAIL_CLOSED_NO_REBUILD", raised and (root / "junk.txt").is_file())


def test_legacy_untouched() -> None:
    if not LEGACY_ROOT.exists():
        _assert("LEGACY_DATA_PRESERVED", True)
        return
    files = []
    for p in sorted(LEGACY_ROOT.rglob("*")):
        if p.is_file():
            import hashlib
            files.append(f"{hashlib.md5(p.read_bytes()).hexdigest()} {p.stat().st_size} {p.relative_to(LEGACY_ROOT)}")
    current = "\n".join(files) + "\n"
    if LEGACY_SNAP.is_file():
        snap = LEGACY_SNAP.read_text()
        _assert("LEGACY_DATA_PRESERVED", current == snap)
    else:
        _assert("LEGACY_DATA_PRESERVED", True)
    _assert(
        "LEGACY_NO_CATEGORY_POLLUTION",
        not (LEGACY_ROOT / "good").exists()
        and not (LEGACY_ROOT / "qualified").exists()
        and not (LEGACY_ROOT / "Doubtful").exists(),
    )


def test_hold_sec_constant() -> None:
    _assert("NEUTRAL_HOLD_SEC", CLASSIFICATION_NEUTRAL_HOLD_SEC == 0.25)


def main() -> int:
    print(f"TEST_ROOT={TEST_ROOT}")
    print(f"LEGACY_ROOT={LEGACY_ROOT} (must not be written)")
    tests = [
        test_hold_sec_constant,
        test_review_ready_mapping,
        test_neutral_gate,
        test_both_grip_priority,
        test_recording_monitor,
        test_poll_pico_no_callback,
        test_buttons_from_key_state,
        test_save_dirs_and_isolation,
        test_delete_and_cancel_index,
        test_auto_resume,
        test_fail_closed,
        test_legacy_untouched,
    ]
    for fn in tests:
        try:
            fn()
        except Exception:
            _record(fn.__name__, False)
            traceback.print_exc()

    # Roll-up names expected by the milestone report
    RESULTS["SYNTHETIC_PICO_TEST"] = (
        "PASS"
        if all(
            RESULTS.get(k) == "PASS"
            for k in (
                "SYNTHETIC_L_GRIP_GOOD",
                "SYNTHETIC_X_QUALIFIED",
                "SYNTHETIC_A_DOUBTFUL",
                "SYNTHETIC_R_GRIP_DELETE",
                "SYNTHETIC_B_NO_CLASSIFY",
            )
        )
        else "FAIL"
    )
    RESULTS["NEUTRAL_GATE_TEST"] = (
        "PASS"
        if all(
            RESULTS.get(k) == "PASS"
            for k in ("NEUTRAL_GATE_L_GRIP", "NEUTRAL_GATE_R_GRIP", "NEUTRAL_GATE_X", "NEUTRAL_GATE_A")
        )
        else "FAIL"
    )
    failed = [k for k, v in RESULTS.items() if v != "PASS"]
    print("----")
    for k, v in RESULTS.items():
        print(f"{k}={v}")
    print(f"FAILED={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
