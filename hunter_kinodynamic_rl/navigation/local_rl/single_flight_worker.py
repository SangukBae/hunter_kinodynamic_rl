"""Single-flight bounded-timeout worker-thread helper.

Extracted (not reimplemented) from ``nodes/real_policy_node.py``'s own
``_infer_with_timeout_thread``/``_inference_thread`` pattern -- see that
method's docstring for the full, honestly-documented rationale and the
CPython "a thread cannot be forcibly killed" limitation this design accepts
(a genuinely wedged call leaks its worker thread; what this DOES guarantee
is that the CALLER never blocks past ``timeout_sec`` and never starts a
second worker thread while a previous one is still alive). Generalized here
(``Callable[[], T]`` instead of a hardcoded ``agent.select_action`` call) so
``navigation.local_rl.live_gazebo_executor.LiveGazeboLocalExecutor`` can
reuse the IDENTICAL single-flight contract for its own frozen-policy
inference calls instead of re-deriving this thread bookkeeping a second
time.
"""

from __future__ import annotations

import threading
from typing import Callable, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class SingleFlightThreadWorker(Generic[T]):
    """One instance per logical call site (e.g. one per executor). Never
    shared between two unrelated bounded calls -- ``busy`` and the internal
    thread handle are per-instance state, exactly like
    ``RealPolicyNode._inference_thread``."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None

    @property
    def busy(self) -> bool:
        """True while a PREVIOUS call's worker thread is still running past
        its own timeout budget -- the caller must not invoke :meth:`call`
        again until this is False (mirrors ``real_policy_node.py``'s
        single-flight guard: never spawn a second worker thread while one
        is already outstanding, which would otherwise leak one new thread
        per tick for as long as a genuine hang lasts)."""
        return self._thread is not None and self._thread.is_alive()

    def call(self, fn: Callable[[], T], timeout_sec: float) -> Tuple[Optional[T], Optional[BaseException], bool]:
        """Runs ``fn()`` on a worker thread, joined with a bounded
        ``timeout_sec``. Returns ``(result, error, timed_out)``:

        - ``(result, None, False)`` -- ``fn()`` returned normally.
        - ``(None, error, False)`` -- ``fn()`` raised; ``error`` is the
          exception instance (never re-raised into the caller's own
          thread/control loop).
        - ``(None, None, True)`` -- exceeded ``timeout_sec`` (or a PRIOR
          call from this same instance is still outstanding -- single-
          flight: no second worker thread is ever started while one is
          alive). The leaked thread (if genuinely hung, not just slow)
          keeps running in the background; a later :meth:`call` will keep
          returning ``timed_out=True`` for as long as it stays alive.
        """
        if self.busy:
            return None, None, True
        box: dict = {}

        def _run() -> None:
            try:
                box["result"] = fn()
            except Exception as e:  # noqa: BLE001 -- must never propagate into the caller's control loop
                box["error"] = e

        thread = threading.Thread(target=_run, daemon=True, name="single_flight_worker")
        self._thread = thread
        thread.start()
        thread.join(timeout_sec)
        if thread.is_alive():
            return None, None, True
        self._thread = None
        if "error" in box:
            return None, box["error"], False
        return box.get("result"), None, False
