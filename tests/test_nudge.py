"""NUDGE: pattern advances only on stalls during insertion. / 삽입 정체 시에만 패턴이 진행."""

import numpy as np
import pytest

from factor.planning.nudge import NudgeSearch


def test_idle_outside_insertion():
    nudge = NudgeSearch(amplitude=0.0015, stall_steps=3)
    for _ in range(10):
        off = nudge.update(0.1, inserting=False)
    assert np.allclose(off, 0.0)


def test_advances_after_stall():
    nudge = NudgeSearch(amplitude=0.0015, stall_steps=3)
    nudge.update(0.070, inserting=True)
    for _ in range(3):
        off = nudge.update(0.070, inserting=True)
    assert np.linalg.norm(off) == pytest.approx(0.0015 / 6, rel=1e-3)  # first spiral point
    for _ in range(3):
        off = nudge.update(0.070, inserting=True)
    assert np.linalg.norm(off) == pytest.approx(2 * 0.0015 / 6, rel=1e-3)


def test_progress_resets_stall():
    nudge = NudgeSearch(amplitude=0.0015, stall_steps=3)
    height = 0.070
    nudge.update(height, inserting=True)
    for _ in range(20):
        height -= 0.001
        off = nudge.update(height, inserting=True)
    assert np.allclose(off, 0.0)


def test_retry_after_pattern_exhausted():
    from factor.planning.nudge import PATTERN

    nudge = NudgeSearch(amplitude=0.0015, stall_steps=1)
    for _ in range(len(PATTERN) - 1):  # every stalled update advances one entry
        nudge.update(0.070, inserting=True)
        assert not nudge.retry
    off = nudge.update(0.070, inserting=True)  # past the last entry: wrap and ask for a retry
    assert nudge.retry and np.allclose(off, 0.0)
    nudge.clear_retry()
    assert not nudge.retry
