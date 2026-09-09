"""G1 OmniPicker post-episode classification review.

Pico raw state is polled without actuating robot controllers. Classification
keys are only accepted after a neutral hold, and only on a new rising edge.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable


CLASSIFICATION_NEUTRAL_HOLD_SEC = 0.25

GOOD = "GOOD"
QUALIFIED = "QUALIFIED"
DOUBTFUL = "DOUBTFUL"
DELETE = "DELETE"
SHUTDOWN = "SHUTDOWN"

CATEGORY_DIR = {
    GOOD: "good",
    QUALIFIED: "qualified",
    DOUBTFUL: "Doubtful",
}

CATEGORY_LABEL = {
    GOOD: "好",
    QUALIFIED: "合格",
    DOUBTFUL: "存疑",
    DELETE: "删除",
}

CLASSIFICATION_BANNER = """
============================================================
                  数据初次分类
============================================================

默认：[好]

左 Grip   → 好      → good
X         → 合格    → qualified
A         → 存疑    → Doubtful
右 Grip   → 删除    → 不保存

------------------------------------------------------------
录制过程中：
右 Grip   → 立即取消本集
============================================================
"""

READY_HINT = ">>> 左Grip=好 / X=合格 / A=存疑 / 右Grip=删除"
NEUTRAL_HINT = "请先松开按键..."


@dataclass
class PicoButtons:
    l_grip: bool = False
    r_grip: bool = False
    x: bool = False
    a: bool = False
    b: bool = False

    @property
    def both_grip(self) -> bool:
        return bool(self.l_grip) and bool(self.r_grip)

    def classification_held(self) -> bool:
        return bool(self.l_grip or self.r_grip or self.x or self.a)


def buttons_from_key_state(key_state: dict | None) -> PicoButtons:
    if not key_state:
        return PicoButtons()
    left = key_state.get("leftHand") or {}
    right = key_state.get("rightHand") or {}
    return PicoButtons(
        l_grip=bool(left.get("gripButtonPressed")),
        r_grip=bool(right.get("gripButtonPressed")),
        x=bool(left.get("primaryButtonPressed")),
        a=bool(right.get("primaryButtonPressed")),
        b=bool(right.get("secondaryButtonPressed")),
    )


def poll_pico_raw(pico_joystick) -> PicoButtons:
    """Refresh raw Pico state without firing bound controller callbacks.

    ``PicoJoystick.update(keys)`` only invokes callbacks for the given keys.
    ``update([])`` therefore does not actuate the robot. Raw TCP state is
    already written by the Pico server thread; ``get_key_state()`` reads it.
    """
    try:
        pico_joystick.update([])
    except Exception:
        pass
    try:
        return buttons_from_key_state(pico_joystick.get_key_state())
    except Exception:
        return PicoButtons()


def grip_monitor_decision(
    session_phase: str,
    l_grip: bool,
    r_grip: bool,
    prev_both: bool,
    prev_r_only: bool,
) -> str | None:
    """Decide what the background monitor may do.

    REVIEW right-grip is owned by classification and must not be treated as a
    RECORDING immediate-discard event.
    """
    both = bool(l_grip) and bool(r_grip)
    r_only = bool(r_grip) and not bool(l_grip)
    if both and not prev_both:
        return SHUTDOWN
    if session_phase == "REVIEW":
        return None
    if r_only and not prev_r_only:
        return "RECORDING_DISCARD"
    return None


class ClassificationReview:
    """Neutral-gated rising-edge classifier for one parked episode."""

    def __init__(
        self,
        hold_sec: float = CLASSIFICATION_NEUTRAL_HOLD_SEC,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.hold_sec = float(hold_sec)
        self.now_fn = now_fn or time.monotonic
        self.phase = "NEUTRAL"
        self._neutral_since: float | None = None
        self._prev = PicoButtons()
        self.became_ready = False

    def feed(self, buttons: PicoButtons) -> str | None:
        self.became_ready = False
        # BOTH_GRIP_SHUTDOWN > RIGHT_GRIP_DELETE > LEFT_GRIP_GOOD
        if buttons.both_grip:
            return SHUTDOWN

        now = self.now_fn()
        if self.phase == "NEUTRAL":
            if buttons.classification_held():
                self._neutral_since = None
            else:
                if self._neutral_since is None:
                    self._neutral_since = now
                elif (now - self._neutral_since) >= self.hold_sec:
                    self.phase = "READY"
                    self._prev = PicoButtons()
                    self.became_ready = True
            return None

        decision = None
        if buttons.r_grip and not self._prev.r_grip:
            decision = DELETE
        elif buttons.l_grip and not self._prev.l_grip:
            decision = GOOD
        elif buttons.x and not self._prev.x:
            decision = QUALIFIED
        elif buttons.a and not self._prev.a:
            decision = DOUBTFUL
        self._prev = PicoButtons(
            l_grip=buttons.l_grip,
            r_grip=buttons.r_grip,
            x=buttons.x,
            a=buttons.a,
            b=buttons.b,
        )
        return decision


def derive_repo_id(base_repo_id: str, category: str) -> str:
    suffix = {
        GOOD: "good",
        QUALIFIED: "qualified",
        DOUBTFUL: "doubtful",
    }[category]
    return f"{base_repo_id}_{suffix}"
