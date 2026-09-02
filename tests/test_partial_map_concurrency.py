"""Requirement K: concurrent access to PartialMap -- one writer thread
(mimicking a scan callback) and one reader thread (mimicking a policy/
inference thread calling channels()/snapshot()) running simultaneously must
never crash, deadlock, or observe a torn/inconsistent array (PartialMap
already guards every public method with a single ``threading.RLock`` --
this test exercises that guarantee under REAL concurrent load rather than
just reading the source)."""

import threading
import time

import numpy as np

from hunter_kinodynamic_rl.config.schema import MappingConfig
from hunter_kinodynamic_rl.navigation.mapping.partial_map import PartialMap


def _make_map(size_cells=64):
    return PartialMap(MappingConfig(mission_size_cells=size_cells), size_cells=size_cells)


def test_concurrent_integrate_scan_and_channels_never_crashes():
    partial_map = _make_map()
    stop = threading.Event()
    errors = []

    def _writer():
        rng = np.random.RandomState(0)
        step = 0
        while not stop.is_set():
            x, y = rng.uniform(-2.0, 2.0, size=2)
            n = 32
            angles = np.linspace(-np.pi, np.pi, n)
            ranges = rng.uniform(0.5, 3.0, size=n)
            try:
                partial_map.integrate_scan((float(x), float(y)), angles, ranges, range_max=5.0, range_min=0.05)
                partial_map.record_visit(float(x), float(y), step)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return
            step += 1

    def _reader():
        while not stop.is_set():
            try:
                channels = partial_map.channels()
                snapshot = partial_map.snapshot()
                # Consistency check: occupied/free/observed_uncertain/unknown
                # must partition every cell exactly once, even mid-write --
                # the lock means every channels() call sees ONE atomic state,
                # never a half-written array.
                total = (
                    channels.occupied.astype(np.int64) + channels.free.astype(np.int64)
                    + channels.unknown.astype(np.int64) + channels.observed_uncertain.astype(np.int64)
                )
                assert np.all(total == 1), "channel partition invariant violated under concurrent access"
                assert snapshot.observed.shape == channels.occupied.shape
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return

    threads = [threading.Thread(target=_writer), threading.Thread(target=_reader), threading.Thread(target=_reader)]
    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "thread did not terminate -- possible deadlock"

    assert not errors, f"concurrent access raised: {errors}"


def test_concurrent_writes_from_two_threads_never_corrupt_visit_counts():
    partial_map = _make_map()
    stop = threading.Event()
    errors = []

    def _writer(offset):
        step = 0
        while not stop.is_set():
            try:
                partial_map.record_visit(float(offset), 0.0, step)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
                return
            step += 1

    threads = [threading.Thread(target=_writer, args=(0.1 * i,)) for i in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert not errors, f"concurrent writes raised: {errors}"
    # No crash + no exception is the core guarantee here; visited_count
    # itself is allowed to saturate/clip per PartialMap's own documented
    # contract, just never go negative or overflow silently.
    channels = partial_map.channels()
    assert np.all(channels.visited >= 0.0)
