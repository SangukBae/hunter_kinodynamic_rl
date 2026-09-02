"""``SingleFlightThreadWorker`` (item 6) -- bounded-timeout, single-flight
worker-thread helper reused by ``LiveGazeboLocalExecutor.run_option`` for
policy inference. Pure Python, no ROS needed."""

from __future__ import annotations

import threading

from hunter_kinodynamic_rl.navigation.local_rl.single_flight_worker import SingleFlightThreadWorker


def test_normal_call_returns_result():
    worker = SingleFlightThreadWorker()
    result, error, timed_out = worker.call(lambda: 42, timeout_sec=1.0)
    assert result == 42
    assert error is None
    assert timed_out is False
    assert worker.busy is False


def test_call_that_raises_reports_error_never_propagates():
    worker = SingleFlightThreadWorker()

    def _boom():
        raise ValueError("kaboom")

    result, error, timed_out = worker.call(_boom, timeout_sec=1.0)
    assert result is None
    assert isinstance(error, ValueError)
    assert timed_out is False
    assert worker.busy is False


def test_slow_call_times_out_and_reports_busy():
    worker = SingleFlightThreadWorker()
    release = threading.Event()

    def _slow():
        release.wait(5.0)
        return "late"

    result, error, timed_out = worker.call(_slow, timeout_sec=0.05)
    assert result is None
    assert error is None
    assert timed_out is True
    assert worker.busy is True  # the orphaned thread is still alive

    # Single-flight: a second call while the first is still outstanding
    # must NOT start a new worker thread -- immediate (None, None, True).
    result2, error2, timed_out2 = worker.call(lambda: "should not run", timeout_sec=0.05)
    assert (result2, error2, timed_out2) == (None, None, True)

    release.set()  # let the orphaned thread finish so it doesn't leak past the test


def test_busy_property_false_before_any_call():
    worker = SingleFlightThreadWorker()
    assert worker.busy is False


def test_sequential_calls_after_completion_are_independent():
    worker = SingleFlightThreadWorker()
    r1, _, t1 = worker.call(lambda: 1, timeout_sec=1.0)
    r2, _, t2 = worker.call(lambda: 2, timeout_sec=1.0)
    assert (r1, t1) == (1, False)
    assert (r2, t2) == (2, False)
