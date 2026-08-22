import numpy as np
import pytest

from map_data.pathsolver import replan as replan_mod
from map_data.pathsolver.replan import ReplanPath, cancel_replan_backend


class Args:
    def __init__(self):
        self.low = (0, 0)
        self.high = (10, 10)
        self.cell_size = 0.5
        self.simplify_path = True
        self.inflate_obstacles = 0.0


@pytest.fixture
def transfer_id():
    """
    A unique transfer id, guaranteed to be removed from the module-level
    cancellation set after the test so state cannot leak between tests.
    """
    tid = "test-replan-cancel-id"
    yield tid
    replan_mod._discard_cancelled(tid)


def _make_replanner(transfer_id=None):
    replanner = ReplanPath(Args(), [], transfer_id=transfer_id)
    # Pre-warm the grid cache with an all-free grid (same pattern as test_astar)
    replanner._reshaped_grid_cache = np.zeros((20, 20), dtype=float)
    return replanner


# ── cancellation helpers ─────────────────────────────────────────────────────


def test_cancel_helpers_roundtrip(transfer_id):
    assert not replan_mod._is_cancelled(transfer_id)

    cancel_replan_backend(transfer_id)
    assert replan_mod._is_cancelled(transfer_id)

    replan_mod._discard_cancelled(transfer_id)
    assert not replan_mod._is_cancelled(transfer_id)


def test_cancel_helpers_ignore_none():
    # None/empty transfer ids are no-ops: nothing is registered, nothing matches
    cancel_replan_backend(None)
    assert not replan_mod._is_cancelled(None)
    replan_mod._discard_cancelled(None)  # must not raise


# ── replan() cancellation contract ───────────────────────────────────────────


def test_replan_completes_without_cancel(transfer_id):
    replanner = _make_replanner(transfer_id)
    path = np.array([[1.0, 1.0], [9.0, 9.0]])

    result = replanner.replan(path)

    assert result is not None
    assert np.allclose(result[0], [1.0, 1.0])
    assert np.allclose(result[-1], [9.0, 9.0])


def test_replan_cancelled_before_start_returns_none(transfer_id):
    replanner = _make_replanner(transfer_id)
    cancel_replan_backend(transfer_id)

    result = replanner.replan(np.array([[1.0, 1.0], [9.0, 9.0]]))

    assert result is None
    # The pending cancellation was consumed when the replan aborted
    assert not replan_mod._is_cancelled(transfer_id)


def test_replan_cancelled_mid_run_stops_and_consumes_cancel(transfer_id):
    replanner = _make_replanner(transfer_id)
    path = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])

    # Inject the cancel deterministically between loop iterations: the
    # collision check runs once per segment, so cancelling from inside it
    # means the cancel lands while segment 0 is being processed.
    calls = []

    def cancelling_colides(path_seg):
        calls.append(path_seg)
        cancel_replan_backend(transfer_id)
        return False

    replanner._colides = cancelling_colides

    result = replanner.replan(path)

    assert result is None
    # Cancelled right after the first segment; the second was never processed
    assert len(calls) == 1
    # The cancel id was consumed (discarded) when the cancellation was honoured
    assert not replan_mod._is_cancelled(transfer_id)
