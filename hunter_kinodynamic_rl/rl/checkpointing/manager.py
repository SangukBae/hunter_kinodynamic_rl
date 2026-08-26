"""Checkpoint save/load for TQC-family agents (section 39): actor, reward
critics, target critics, optional risk critic, optimizers, entropy
coefficient, training step, RNG seed, and a config snapshot -- all in one
``.pt`` file plus a sibling ``.json`` manifest for the non-tensor metadata
(human-readable, diffable across runs), PLUS (section item-3) the paired
replay buffer, all three published as one atomic, hash/generation-verified
unit.

Operates on any agent object exposing the ``CHECKPOINT_COMPONENTS`` protocol
below (a dict of {name: nn.Module-or-Optimizer-or-None}) rather than a fixed
class, so :mod:`rl.algorithms.tqc.agent` (no risk critic) and
:mod:`rl.algorithms.kinodynamic_tqc.agent` (adds one) share this exact
save/load code -- the manifest simply records which components were present.

section item-3 (checkpoint generation/atomicity): TWO checkpoint layouts
coexist, on purpose, never silently interchangeable:

- **generation layout** (:func:`save_generation`/:func:`load_generation`,
  the layout every production call site in this package uses): ``<tag>``
  under ``directory`` is a SYMLINK to ``.generations/<generation>/``, a
  freshly-created, uuid-named directory containing ``model.pt``,
  ``manifest.json``, and ``replay.npz`` together. The generation's OWN
  content directory is written to COMPLETELY before anything is published
  -- "publishing" is a single atomic symlink rename
  (:func:`_publish_generation_symlink`) at the very end, so any reader
  either sees the complete PREVIOUS generation or the complete NEW one,
  NEVER a partial mix. The SAME ``generation`` id is embedded in all THREE
  files (``model.pt``'s own payload, ``manifest.json``, and
  ``replay.npz``'s own field) and cross-checked on every load -- swapping
  in just one file from a DIFFERENT generation (even if that file is
  itself perfectly valid on its own) is detected and rejected, not just a
  hash mismatch on one pairwise comparison.
- **legacy flat layout** (:func:`save_legacy_flat`/:func:`load_legacy_flat`,
  the ORIGINAL pre-item-3 layout: ``<tag>.pt``/``<tag>.json`` as plain
  sibling files directly under ``directory``, replay saved separately by
  the caller) -- kept ONLY for reading checkpoints that already exist on
  disk from before this layout existed. Never used to WRITE a new
  checkpoint from production code; a caller must explicitly ask for it by
  name, never a silent fallback.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional


def sha256_of_file(path: str) -> str:
    """Used to embed (at save time) and re-verify (at load time) a
    checkpoint SET's cross-file consistency -- see ``save_generation``/
    ``load_generation``'s docstrings."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_path(path: str) -> None:
    """section item-3 (round 2): flushes ``path`` (file OR directory) to
    stable storage -- ``torch.save``/``ReplayBuffer.save`` write via their
    own tmp-then-``os.replace`` sequence but do not themselves ``fsync``,
    so a crash immediately after either call could still lose the write
    even though it's already "on disk" from the OS's point of view (dirty
    page cache, not yet flushed). Directory fsyncs are needed too, on
    POSIX, to persist the fact that a NEW directory entry (a file, or the
    published symlink itself) now exists -- ``os.replace`` alone only
    guarantees the rename is atomic, not that it survives a power loss
    before the containing directory's own metadata is flushed."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# section item-3 (checkpoint-prune-safety fix): per-generation advisory
# (POSIX ``fcntl.flock``) locking -- closes the TOCTOU between
# :func:`prune_orphan_generations`' final "is this generation still
# referenced" check and its ``shutil.rmtree`` call, WITHOUT a full
# lock/lease service: :func:`load_generation` holds a SHARED lock on a
# generation's own ``.lock`` file for the ENTIRE duration it reads
# ``model.pt``/``manifest.json``/``replay.npz`` from it; prune only ever
# deletes a generation after successfully acquiring an EXCLUSIVE,
# NON-BLOCKING lock on that SAME file immediately before ``rmtree`` --
# the OS makes that acquisition atomic with respect to any concurrent
# shared holder, so there is no window in which a load already in flight
# can have its generation directory pulled out from under it. Unlike the
# rest of this package's ROS/Gazebo I/O (which must never block, since a
# hung service could hang forever), a local ``flock`` held only for the
# duration of a small-directory ``rmtree`` is bounded and safe to block
# briefly on -- so :func:`load_generation`'s acquisition is a normal
# blocking ``flock`` (never a network call), while prune's is
# non-blocking and simply skips (never deletes) a generation it can't
# lock, leaving it for a later call to retry.
_GENERATION_LOCK_FILENAME = ".lock"
_PRUNE_LOCK_FILENAME = ".prune.lock"
_LIFECYCLE_LOCK_FILENAME = ".generation-lifecycle.lock"


class PruneAlreadyRunningError(RuntimeError):
    """Raised by :func:`prune_orphan_generations` when another prune call
    already holds the directory-level exclusive lock -- section item-3's
    "enforce non-concurrent execution in code" requirement: rather than a
    full lock/lease SERVICE, two prune invocations against the SAME
    ``directory`` are simply forbidden from ever running concurrently,
    enforced by a single ``.prune.lock`` file each call must exclusively
    (non-blockingly) acquire for its ENTIRE run."""


class PrunePartialFailureError(RuntimeError):
    """Raised by :func:`prune_orphan_generations` when at least one
    eligible generation's ``shutil.rmtree`` genuinely failed (permissions,
    a file busy, disk error, ...) -- section item-3: a failed deletion
    must never be silently treated as "deleted" (the pre-fix behavior,
    ``ignore_errors=True``). ``.deleted`` carries the generations that
    WERE actually removed this call (never includes a failed one);
    ``.failed`` carries ``{generation_id: repr(exception)}`` for the
    ones that weren't."""

    def __init__(self, deleted: List[str], failed: Dict[str, str]):
        super().__init__(
            f"prune_orphan_generations: failed to delete {sorted(failed)} (deleted {sorted(deleted)} "
            f"successfully) -- see .failed for the per-generation error"
        )
        self.deleted = deleted
        self.failed = failed


@contextlib.contextmanager
def _flock(path: str, *, exclusive: bool, blocking: bool):
    """Opens (creating if needed) ``path`` and holds a POSIX advisory lock
    on it for the duration of the ``with`` block. Yields ``True`` if the
    lock was acquired, ``False`` if ``blocking=False`` and it could not be
    (caller decides what "couldn't lock it" means -- for prune, "skip this
    candidate"; never raises for that case, only for a genuine OSError
    opening/closing the lock file itself)."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            op |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, op)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _state_dict_or_none(component) -> Optional[dict]:
    if component is None:
        return None
    return component.state_dict()


def _resolve_missing(components: Dict[str, Any], payload: Dict[str, Any], allow_missing: Iterable[str]):
    """Shared by both layouts' load: returns (loaded_names, skipped_names,
    missing_required_names) -- does NOT itself load state_dicts (the
    caller does that, since the two layouts fetch ``payload`` differently)."""
    allow_missing = set(allow_missing)
    loaded, skipped, missing_required = [], [], []
    for name, comp in components.items():
        state = payload.get(name)
        if comp is None or state is None:
            skipped.append(name)
            if comp is not None and name not in allow_missing:
                missing_required.append(name)
            continue
        loaded.append(name)
    return loaded, skipped, missing_required


def _raise_if_missing_required(pt_path: str, missing_required):
    if missing_required:
        raise RuntimeError(
            f"checkpoint {pt_path} is missing required component(s) {sorted(missing_required)} -- "
            "this agent's architecture declares them (checkpoint_components()) but the checkpoint's "
            "own payload doesn't have them. Refusing to continue with a freshly-(randomly-)initialised "
            "stand-in for a component training/evaluation/deployment actually needs. If this is a "
            "DELIBERATE warm-start from a differently-configured checkpoint, pass "
            f"allow_missing={sorted(missing_required)!r} explicitly."
        )


# ------------------------------------------------------- generation layout
def _publish_generation_symlink(directory: str, tag: str, generation: str) -> None:
    """The ONE atomic step that makes a fully-written generation visible at
    ``directory/<tag>`` -- a RELATIVE symlink (so the whole ``directory``
    tree stays relocatable/copyable as a unit) swapped into place via
    ``os.replace`` on the symlink's own path (POSIX ``rename()`` on a
    symlink path atomically repoints it, whether or not something was
    already there -- this is what makes the publish atomic, not the
    symlink mechanism per se).

    section item-3 (round 2, found via the concurrency stress test):
    ``tmp_link`` is named using ``generation`` (already a fresh, unique
    uuid4 per :func:`save_generation` call), NOT a fixed
    ``.{tag}.symlink.tmp`` name -- the fixed name was a genuine race: two
    THREADS/PROCESSES concurrently publishing to the SAME ``tag`` could
    both pass the ``os.path.lexists`` check before either created the
    file, then both call ``os.symlink`` on the identical path, and the
    second one raises ``FileExistsError`` (``os.symlink`` refuses to
    overwrite an existing path, unlike ``os.replace``). A per-generation
    unique staging path removes the collision entirely; the final
    ``os.replace(tmp_link, tag_path)`` step is still atomic per POSIX
    ``rename()``, so two concurrent publishes to the same tag simply
    resolve to ordinary last-writer-wins (never a crash, never a partial/
    corrupt symlink)."""
    tag_path = os.path.join(directory, tag)
    tmp_link = os.path.join(directory, f".{tag}.{generation}.symlink.tmp")
    if os.path.lexists(tmp_link):
        os.remove(tmp_link)
    os.symlink(os.path.join(".generations", generation), tmp_link)
    os.replace(tmp_link, tag_path)
    # section item-3 (round 2): persist the rename itself -- see
    # _fsync_path's own docstring for why this is needed on top of
    # os.replace's atomicity guarantee.
    _fsync_path(directory)


def save_generation(
    directory: str, tag: str, components: Dict[str, Any], meta: Dict[str, Any],
    replay_buffer, generation: Optional[str] = None,
) -> str:
    """Atomically publishes a COMPLETE checkpoint generation -- model,
    manifest, AND the paired replay buffer -- as one unit (section item-3).
    See this module's docstring for the staging-directory + atomic-symlink-
    rename mechanism. Returns the ``generation`` id used (a fresh uuid4 hex
    if not supplied).

    Holds a directory-level SHARED lifecycle lock before ``gen_dir`` is
    created and until AFTER publication, plus an EXCLUSIVE advisory lock
    on this generation's own ``.lock`` file from the moment ``gen_dir`` is
    created until AFTER
    :func:`_publish_generation_symlink` returns -- a fresh, freshly-created
    generation directory is, by construction, unreferenced by any tag until
    publish, which otherwise makes it indistinguishable from a genuine
    orphan to :func:`prune_orphan_generations`' scan (see that function's
    own docstring for the mark-then-sweep window): a concurrent prune run
    with a small enough ``min_age_sec`` could mark this in-progress
    generation orphaned and, on a LATER call once that marker aged out,
    ``rmtree`` it out from under this still-running save, before the tag
    was ever published. Blocking (not non-blocking): a fresh uuid4
    directory is never contended by anything except prune's own
    non-blocking attempt, so this can only ever wait on a prune call that
    is about to lose the race anyway, never deadlock (prune's per-
    generation acquisition is always non-blocking; see ``_flock``'s
    docstring)."""
    generation = generation or uuid.uuid4().hex
    os.makedirs(directory, exist_ok=True)
    lifecycle_lock_path = os.path.join(directory, _LIFECYCLE_LOCK_FILENAME)
    # Lock order is lifecycle -> generation. Prune takes prune -> lifecycle
    # -> generation, and generation acquisition on its scan/delete path is
    # non-blocking. Loads take only a generation lock. This ordering has no
    # cycle and lets independent saves proceed concurrently under SHARED
    # lifecycle locks while excluding prune from the mkdir-to-publish gap.
    with _flock(lifecycle_lock_path, exclusive=False, blocking=True):
        return _save_generation_while_prune_excluded(
            directory, tag, components, meta, replay_buffer, generation)


def _save_generation_while_prune_excluded(
    directory: str, tag: str, components: Dict[str, Any], meta: Dict[str, Any], replay_buffer,
    generation: str,
) -> str:
    """Implementation of :func:`save_generation` while its shared
    directory-lifecycle lock excludes prune from the complete
    create/write/publish interval."""
    import torch

    generations_dir = os.path.join(directory, ".generations")
    os.makedirs(generations_dir, exist_ok=True)
    gen_dir = os.path.join(generations_dir, generation)
    os.makedirs(gen_dir, exist_ok=False)  # a fresh uuid must never already exist

    lock_path = os.path.join(gen_dir, _GENERATION_LOCK_FILENAME)
    with _flock(lock_path, exclusive=True, blocking=True):
        pt_path = os.path.join(gen_dir, "model.pt")
        payload = {name: _state_dict_or_none(comp) for name, comp in components.items()}
        payload["__generation__"] = generation
        torch.save(payload, pt_path)
        _fsync_path(pt_path)  # section item-3 (round 2): see _fsync_path's docstring
        pt_sha256 = sha256_of_file(pt_path)
        pt_size_bytes = os.path.getsize(pt_path)

        replay_path = os.path.join(gen_dir, "replay.npz")
        replay_buffer.save(replay_path, generation=generation)
        _fsync_path(replay_path)
        replay_sha256 = sha256_of_file(replay_path)
        replay_size_bytes = os.path.getsize(replay_path)

        manifest = dict(meta)
        manifest.update({
            "checkpoint_layout": "generation_v1",
            "generation": generation,
            "pt_sha256": pt_sha256, "pt_size_bytes": pt_size_bytes,
            "replay_sha256": replay_sha256, "replay_size_bytes": replay_size_bytes,
            "present_components": sorted(name for name, comp in components.items() if comp is not None),
        })
        manifest_path = os.path.join(gen_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        # Directory-entry durability for the 3 files just created inside
        # gen_dir -- see _fsync_path's own docstring.
        _fsync_path(gen_dir)
        # section item-5 (round 3): the PARENT `.generations` directory's own
        # entry (this generation's freshly-created subdirectory) also needs
        # its own fsync -- without it, a crash could persist gen_dir's own
        # contents durably (already fsynced above) while `.generations` itself
        # forgets the new subdirectory ever existed, leaving an orphaned inode
        # unreachable from any listing. Done here, AFTER the generation is
        # fully written but BEFORE publish (the symlink swap below), matching
        # the explicit ordering requested.
        _fsync_path(generations_dir)

        _publish_generation_symlink(directory, tag, generation)
    return generation


_ORPHAN_MARKER_FILENAME = ".orphaned_since"


def prune_orphan_generations(directory: str, *, keep_last_n: int = 3, min_age_sec: float = 3600.0) -> List[str]:
    """section item-3 (round 2)/item-5 (round 3): an EXPLICIT, opt-in
    maintenance utility -- NEVER called automatically by
    :func:`save_generation`/:func:`load_generation`, and never called by
    any training/evaluation/deployment code path in this package -- that
    removes ORPHAN ``.generations/<uuid>/`` directories (ones NO tag
    symlink directly under ``directory`` currently resolves to) that have
    been CONTINUOUSLY orphaned for at least ``min_age_sec`` (default 1
    hour) AND are beyond the ``keep_last_n`` most-recently-orphaned
    survivors (default 3, an extra retention buffer for recent-history
    debugging).

    Handled conservatively, per the explicit ask this section responds to:

    - Re-checks which generations are currently REFERENCED (by any tag
      symlink) at the START of every call, AND again immediately before
      deleting EACH individual candidate -- narrowing the window against a
      concurrent :func:`save_generation` call republishing some tag to
      point at a generation this function had already scanned as an
      orphan.
    - **section item-5 (round 3) fix**: the round-2 version measured an
      orphan's "age" from its ON-DISK CREATION time (``os.path.getmtime``)
      -- a genuine concurrent-delete race: a generation that was
      REFERENCED continuously for hours (e.g. a long-lived checkpoint
      resumed many times) and only became orphaned a MOMENT ago (a
      concurrent :func:`save_generation` just repointed the tag away from
      it) would already look "old enough" by creation time, even though a
      reader could have resolved that SAME tag to this SAME generation
      microseconds earlier and not yet finished reading it. Fixed with a
      MARK-THEN-SWEEP protocol: the FIRST time this function observes a
      generation as orphaned, it writes a persistent marker file
      (``.orphaned_since``, inside that generation's own directory,
      fsynced) recording ``time.time()`` -- and does NOT delete it yet,
      regardless of the generation's own age. Only a LATER call, once
      ``min_age_sec`` has elapsed since THAT marker was written (i.e. the
      generation has been CONTINUOUSLY observed orphaned across at least
      two separate invocations spanning that whole window), will actually
      delete it. A generation that flips back to referenced between two
      calls has its marker removed, restarting the window from scratch if
      it's ever orphaned again. This means a single ad-hoc invocation of
      this function can never delete anything freshly orphaned -- by
      design, a real maintenance workflow (e.g. periodic, via cron) is
      required.
    - **section item-3 (checkpoint-prune-safety fix), on top of the
      mark-then-sweep window above**: THREE further, code-enforced
      protections, since mark-then-sweep alone only narrows the
      concurrent-delete race, it doesn't close it:

      1. **No two prune calls against the same ``directory`` may run
         concurrently at all** -- enforced (never merely documented) by
         acquiring an EXCLUSIVE, non-blocking lock on a directory-level
         ``.prune.lock`` file for this call's ENTIRE run; a second,
         genuinely concurrent call raises :class:`PruneAlreadyRunningError`
         immediately rather than racing the first.
      2. **The final "is this generation still referenced" check
         immediately before deletion is replaced by an actual EXCLUSIVE,
         non-blocking lock acquisition on that generation's own
         ``.lock`` file** (the SAME file :func:`load_generation` holds a
         SHARED lock on for the duration of its own read) -- this is
         atomic with respect to a concurrent load already in flight in a
         way a plain re-check-then-delete never can be: if the lock can't
         be acquired, the generation is left untouched (its marker stays,
         to be retried on a future call) instead of being deleted out
         from under an in-progress reader.
      3. **Save and prune are separated by a directory-level lifecycle
         lock**: saves hold it SHARED from before generation-directory
         creation through publication; prune requires it EXCLUSIVELY for
         its complete scan/mark/sweep. Marker create/read/remove operations
         additionally require the generation's EXCLUSIVE lock. Therefore
         prune cannot mark an unpublished in-flight save as orphaned, nor
         carry that stale marker across a later referenced period.
    - ``.orphaned_since`` is fsynced together with its OWN generation
      directory on every create/update/remove (not just the marker file
      itself), and ``.generations`` (the parent) is fsynced after any
      generation is actually deleted -- durability for the directory-entry
      changes those operations make, matching :func:`save_generation`'s
      own convention (see ``_fsync_path``'s docstring).
    - A ``shutil.rmtree`` failure for one candidate is NEVER silently
      ignored (the pre-fix ``ignore_errors=True``) and never counted as
      "deleted" -- it's collected, and if any candidate failed, this
      function raises :class:`PrunePartialFailureError` at the end
      (``.deleted``/``.failed`` carry exactly which generations were
      actually removed vs. which failed and why); a call where every
      candidate either wasn't eligible or was removed cleanly still just
      returns the list of generation ids actually deleted, as before."""
    generations_dir = os.path.join(directory, ".generations")
    if not os.path.isdir(generations_dir):
        return []

    prune_lock_path = os.path.join(directory, _PRUNE_LOCK_FILENAME)
    with _flock(prune_lock_path, exclusive=True, blocking=False) as acquired:
        if not acquired:
            raise PruneAlreadyRunningError(
                f"prune_orphan_generations({directory!r}) is already running in another process/thread "
                f"(held exclusive lock on {prune_lock_path!r}) -- concurrent prune runs against the same "
                "directory are forbidden; wait for it to finish and retry."
            )
        lifecycle_lock_path = os.path.join(directory, _LIFECYCLE_LOCK_FILENAME)
        # A save holds this lock SHARED from before generation-directory
        # creation through tag publication. If one is active, skip this
        # maintenance pass without touching orphan markers; a later call can
        # retry. The separate prune lock above still distinguishes/forbids
        # two concurrent prune invocations.
        with _flock(lifecycle_lock_path, exclusive=True, blocking=False) as lifecycle_acquired:
            if not lifecycle_acquired:
                return []
            return _prune_orphan_generations_locked(directory, generations_dir, keep_last_n, min_age_sec)


def _currently_referenced_generations(directory: str) -> set:
    referenced = set()
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        if os.path.islink(full):
            referenced.add(os.path.basename(os.path.realpath(full)))
    return referenced


def _prune_orphan_generations_locked(
    directory: str, generations_dir: str, keep_last_n: int, min_age_sec: float,
) -> List[str]:
    """The scan/mark/sweep body of :func:`prune_orphan_generations`, run
    ONLY while this process holds the directory-level exclusive prune
    lock (see caller)."""
    now = time.time()
    eligible = []  # (orphaned_since, name, gen_dir) -- past min_age_sec, not yet deleted
    for name in os.listdir(generations_dir):
        gen_dir = os.path.join(generations_dir, name)
        if not os.path.isdir(gen_dir):
            continue
        lock_path = os.path.join(gen_dir, _GENERATION_LOCK_FILENAME)
        try:
            # Marker state is part of the generation lifecycle too. Never
            # create/read/remove it while a loader holds a shared lease.
            with _flock(lock_path, exclusive=True, blocking=False) as locked:
                if not locked:
                    continue
                marker_path = os.path.join(gen_dir, _ORPHAN_MARKER_FILENAME)
                if name in _currently_referenced_generations(directory):
                    # No longer (or never) orphaned -- clear any stale marker
                    # so a future orphan period starts a fresh grace window.
                    if os.path.isfile(marker_path):
                        os.remove(marker_path)
                        _fsync_path(gen_dir)
                    continue
                if not os.path.isfile(marker_path):
                    # First observation in this continuous orphan period:
                    # mark only, regardless of min_age_sec or creation age.
                    with open(marker_path, "w") as f:
                        f.write(repr(now))
                        f.flush()
                        os.fsync(f.fileno())
                    _fsync_path(gen_dir)
                    continue
                try:
                    with open(marker_path) as f:
                        orphaned_since = float(f.read().strip())
                except (OSError, ValueError):
                    # Corrupt/unreadable means "just orphaned now", never
                    # "old enough": rewrite conservatively under the lock.
                    with open(marker_path, "w") as f:
                        f.write(repr(now))
                        f.flush()
                        os.fsync(f.fileno())
                    _fsync_path(gen_dir)
                    continue
                if now - orphaned_since >= min_age_sec:
                    eligible.append((orphaned_since, name, gen_dir))
        except FileNotFoundError:
            continue

    eligible.sort()  # oldest-orphaned-since first
    to_delete = eligible[:-keep_last_n] if keep_last_n > 0 else eligible

    deleted: List[str] = []
    failed: Dict[str, str] = {}
    for _orphaned_since, name, gen_dir in to_delete:
        if name in _currently_referenced_generations(directory):
            continue
        # section item-3: the ACTUAL TOCTOU fix -- an exclusive,
        # non-blocking lock acquisition on this generation's own .lock
        # file, atomic w.r.t. any concurrent load_generation() holding a
        # shared lock on it. Never deletes if it can't get the lock;
        # never blocks waiting for one either (this candidate is simply
        # retried on a future call, exactly like an unelapsed marker).
        lock_path = os.path.join(gen_dir, _GENERATION_LOCK_FILENAME)
        try:
            with _flock(lock_path, exclusive=True, blocking=False) as locked:
                if not locked:
                    continue
                if name in _currently_referenced_generations(directory):
                    marker_path = os.path.join(gen_dir, _ORPHAN_MARKER_FILENAME)
                    if os.path.isfile(marker_path):
                        os.remove(marker_path)
                        _fsync_path(gen_dir)
                    continue
                try:
                    shutil.rmtree(gen_dir)  # NOT ignore_errors -- a real failure must never look like a deletion
                except FileNotFoundError:
                    pass  # already gone by the time rmtree ran -- the desired end state, not a failure
                except OSError as e:
                    failed[name] = repr(e)
                    continue
        except FileNotFoundError:
            pass  # gen_dir (or its lock file) was already gone by the time we reached it -- already deleted
        deleted.append(name)

    if deleted:
        # section item-3: persist the removal of these directory entries
        # from `.generations` itself -- see _fsync_path's own docstring.
        _fsync_path(generations_dir)

    if failed:
        raise PrunePartialFailureError(deleted=deleted, failed=failed)
    return deleted


def load_generation(directory: str, tag: str, components: Dict[str, Any], map_location=None,
                     *, allow_missing: Iterable[str] = ()) -> Dict[str, Any]:
    """Loads a generation-layout checkpoint (section item-3), verifying:

    1. ``model.pt``'s own embedded ``__generation__`` == ``manifest.json``'s
       recorded ``generation`` == ``replay.npz``'s own embedded
       ``generation`` field -- catches ANY ONE of the three files being
       swapped for a different generation's, even if that file is
       perfectly well-formed on its own.
    2. ``model.pt``'s actual SHA-256/size == what ``manifest.json``
       recorded at save time -- catches corruption or an out-of-band edit
       that happens to keep the SAME (stale) generation tag.
    3. ``replay.npz``'s actual SHA-256/size == what ``manifest.json``
       recorded, the SAME way.

    Raises :class:`RuntimeError` on any mismatch, :class:`FileNotFoundError`
    if the checkpoint (or the required ``replay.npz``) doesn't exist.
    Component-completeness (``allow_missing``) works exactly like
    :func:`load_legacy_flat`.

    section item-3 (round 2, TOCTOU / concurrent-save fix): ``tag`` is
    resolved to its REAL, concrete ``.generations/<uuid>/`` directory
    EXACTLY ONCE, at the very top of this function, via ``os.path.realpath``.
    Every file this function reads below (``model.pt``/``manifest.json``/
    ``replay.npz``), AND the ``replay_path`` handed back to the caller for
    a LATER ``ReplayBuffer.load()`` call, all target this SAME resolved
    directory -- ``tag_path`` (the symlink itself) is never touched again
    after this one resolution. Without this, a CONCURRENT
    :func:`save_generation` call republishing ``tag`` to a DIFFERENT
    generation partway through -- e.g. between this function verifying
    generation A's ``model.pt``/``manifest.json`` and the CALLER later
    opening ``replay_path`` -- could silently mix generations: this
    function's own verification would have genuinely passed against
    generation A, yet the caller's subsequent ``ReplayBuffer.load(replay_path)``
    would transparently read generation B's ``replay.npz`` instead, because
    ``tag_path/replay.npz`` re-traverses the (now repointed) symlink on
    every fresh open. Pinning to the resolved directory closes this
    completely: old generations are never deleted by anything in this
    module except :func:`prune_orphan_generations` (the explicit,
    conservative, never-automatic cleanup utility) -- so a resolved path
    handed out here stays valid and stable for as long as the caller needs
    it, regardless of how many times ``tag`` is republished afterward.

    section item-3 (checkpoint-prune-safety fix): holds a SHARED advisory
    lock on this generation's own ``.lock`` file for the ENTIRE duration
    of the read below -- ``prune_orphan_generations`` only ever deletes a
    generation after acquiring an EXCLUSIVE, non-blocking lock on that
    SAME file, so a load already in flight can never have its generation
    directory removed out from under it (see the lock helpers' own
    module-level docstring for the full TOCTOU rationale).

    **Scope of the lock is exactly this function's own call** -- it is
    released the instant this function returns. A caller that still needs
    to dereference ``result["replay_path"]`` afterward (e.g.
    ``ReplayBuffer.load(result["replay_path"])``, as
    ``trainer_base.py::_resume_from`` does) gets NO protection from THIS
    function once it has returned: a concurrent prune could delete the
    generation in the window between this call returning and the caller's
    own subsequent read. Use :func:`load_generation_lease` instead for
    that case -- it keeps the SAME shared lock held for the caller's
    entire ``with`` block, not just this read."""
    tag_path = os.path.join(directory, tag)
    gen_dir = os.path.realpath(tag_path)
    if not os.path.isdir(gen_dir):
        raise FileNotFoundError(
            f"{tag_path} is not a generation-layout checkpoint directory (missing, not a directory, or a "
            "broken symlink) -- use load_legacy_flat() for a pre-item-3 flat-file checkpoint"
        )
    lock_path = os.path.join(gen_dir, _GENERATION_LOCK_FILENAME)
    with _flock(lock_path, exclusive=False, blocking=True):
        return _read_and_verify_generation(tag_path, gen_dir, components, map_location, allow_missing)


@contextlib.contextmanager
def load_generation_lease(directory: str, tag: str, components: Dict[str, Any], map_location=None,
                           *, allow_missing: Iterable[str] = ()):
    """Like :func:`load_generation`, but keeps the SAME shared per-
    generation advisory lock held for the caller's ENTIRE ``with`` block,
    not just this function's own read of ``model.pt``/``manifest.json``/
    ``replay.npz``'s header.

    This closes a genuine TOCTOU: :func:`load_generation` itself only ever
    ``peek``s ``replay.npz``'s embedded generation id
    (``peek_replay_generation``) -- it never deserializes the buffer's
    actual array payload. That deserialization is the CALLER's job, done
    via ``ReplayBuffer.load(result["replay_path"])`` AFTER
    ``load_generation`` has already returned (and, with that function
    alone, already released its lock). A concurrent
    :func:`prune_orphan_generations` call racing exactly that window --
    the generation just became unreferenced (e.g. a newer save just
    republished the same tag) and was already aged past ``min_age_sec``
    from an earlier orphan-mark -- could delete the generation directory,
    including ``replay.npz``, before the caller's own
    ``ReplayBuffer.load`` call ever runs.

    Callers that need to read anything else derived from the resolved
    generation directory (chiefly ``result["replay_path"]``) MUST do so
    INSIDE this context manager's ``with`` block -- see
    ``trainer_base.py::_resume_from`` for the real, fixed call site."""
    tag_path = os.path.join(directory, tag)
    gen_dir = os.path.realpath(tag_path)
    if not os.path.isdir(gen_dir):
        raise FileNotFoundError(
            f"{tag_path} is not a generation-layout checkpoint directory (missing, not a directory, or a "
            "broken symlink) -- use load_legacy_flat() for a pre-item-3 flat-file checkpoint"
        )
    lock_path = os.path.join(gen_dir, _GENERATION_LOCK_FILENAME)
    with _flock(lock_path, exclusive=False, blocking=True):
        yield _read_and_verify_generation(tag_path, gen_dir, components, map_location, allow_missing)


def _read_and_verify_generation(
    tag_path: str, gen_dir: str, components: Dict[str, Any], map_location, allow_missing: Iterable[str],
) -> Dict[str, Any]:
    """The actual read/verify/load-state_dict body of :func:`load_generation`,
    factored out purely so the shared-lock ``with`` block above wraps a
    single call rather than needing to re-indent this whole function."""
    import torch

    pt_path = os.path.join(gen_dir, "model.pt")
    manifest_path = os.path.join(gen_dir, "manifest.json")
    replay_path = os.path.join(gen_dir, "replay.npz")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(pt_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    if not os.path.isfile(replay_path):
        raise FileNotFoundError(
            f"incomplete checkpoint generation: {pt_path} exists but its companion {replay_path} is "
            "missing -- refusing to resume/evaluate with a silently-absent replay buffer"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    recorded_generation = manifest.get("generation")

    payload = torch.load(pt_path, map_location=map_location, weights_only=True)
    embedded_generation = payload.pop("__generation__", None)

    from hunter_kinodynamic_rl.rl.replay.buffer import peek_replay_generation
    replay_generation = peek_replay_generation(replay_path)

    if not (embedded_generation == recorded_generation == replay_generation):
        raise RuntimeError(
            f"checkpoint generation mismatch at {tag_path}: model.pt embeds generation="
            f"{embedded_generation!r}, manifest.json records generation={recorded_generation!r}, "
            f"replay.npz embeds generation={replay_generation!r} -- these must all be IDENTICAL. "
            "This means one of the three files was replaced independently of the other two (e.g. an "
            "out-of-band file copy/swap) -- refusing to resume/evaluate/deploy against a mismatched "
            "checkpoint set."
        )

    # section item-5 (round 3): the SIZE fields were recorded in the
    # manifest and this docstring's own ITEM 2 above claimed they were
    # verified -- but the code never actually checked them, only the
    # sha256 hashes (a real, if narrow, false-guarantee gap: a sha256
    # match already cryptographically implies a size match for any
    # realistic corruption, but the manifest/docstring claimed an
    # INDEPENDENT check that was never actually made). Checked FIRST
    # (cheap `os.path.getsize`, no full-file read) so an obviously
    # truncated/extended file fails fast with a size-specific message
    # before paying for a full sha256 recompute.
    actual_pt_size_bytes = os.path.getsize(pt_path)
    recorded_pt_size_bytes = manifest.get("pt_size_bytes")
    if recorded_pt_size_bytes is not None and actual_pt_size_bytes != recorded_pt_size_bytes:
        raise RuntimeError(
            f"checkpoint set inconsistency at {tag_path}: model.pt's actual size={actual_pt_size_bytes} bytes "
            f"does not match manifest.json's recorded pt_size_bytes={recorded_pt_size_bytes} -- the file was "
            "truncated, extended, or otherwise corrupted after being saved."
        )
    actual_pt_sha256 = sha256_of_file(pt_path)
    recorded_pt_sha256 = manifest.get("pt_sha256")
    if recorded_pt_sha256 and actual_pt_sha256 != recorded_pt_sha256:
        raise RuntimeError(
            f"checkpoint set inconsistency at {tag_path}: model.pt's actual sha256={actual_pt_sha256} "
            f"does not match manifest.json's recorded pt_sha256={recorded_pt_sha256} -- the file was "
            "corrupted or edited out-of-band after being saved."
        )
    actual_replay_size_bytes = os.path.getsize(replay_path)
    recorded_replay_size_bytes = manifest.get("replay_size_bytes")
    if recorded_replay_size_bytes is not None and actual_replay_size_bytes != recorded_replay_size_bytes:
        raise RuntimeError(
            f"checkpoint set inconsistency at {tag_path}: replay.npz's actual size="
            f"{actual_replay_size_bytes} bytes does not match manifest.json's recorded "
            f"replay_size_bytes={recorded_replay_size_bytes} -- the file was truncated, extended, or "
            "otherwise corrupted after being saved."
        )
    actual_replay_sha256 = sha256_of_file(replay_path)
    recorded_replay_sha256 = manifest.get("replay_sha256")
    if recorded_replay_sha256 and actual_replay_sha256 != recorded_replay_sha256:
        raise RuntimeError(
            f"checkpoint set inconsistency at {tag_path}: replay.npz's actual sha256="
            f"{actual_replay_sha256} does not match manifest.json's recorded replay_sha256="
            f"{recorded_replay_sha256} -- the file was corrupted or edited out-of-band after being saved."
        )

    loaded, skipped, missing_required = _resolve_missing(components, payload, allow_missing)
    _raise_if_missing_required(pt_path, missing_required)
    for name in loaded:
        components[name].load_state_dict(payload[name])

    return {
        "manifest": manifest, "loaded": sorted(loaded), "skipped": sorted(skipped),
        "generation": recorded_generation, "replay_path": replay_path,
        # section item-3 (round 2): the RESOLVED, concrete generation
        # directory this load actually read from -- `replay_path` is
        # already inside it, but callers implementing their own tooling
        # (e.g. prune_orphan_generations' own "is this generation still
        # referenced" check) never need to re-resolve `tag` themselves.
        "generation_dir": gen_dir,
    }


# ------------------------------------------------------------ legacy flat layout
def save_legacy_flat(directory: str, filename: str, components: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """section item-3: the ORIGINAL (pre-generation-layout) save -- kept
    ONLY so existing on-disk checkpoints written by earlier code stay
    readable via :func:`load_legacy_flat`. No production call site writes
    new checkpoints this way anymore; every one of them uses
    :func:`save_generation`, which additionally publishes the paired
    replay buffer atomically and cross-verifies generation/hash on load.

    ATOMIC per-file only (not as a set): both the ``.pt`` and ``.json`` are
    written to a ``.tmp`` sibling first, then ``os.replace()``'d into
    place -- a crash/kill mid-save can never leave a half-written,
    unreadable checkpoint at the real path, but a REPLAY file saved
    separately by the caller (the old convention) is NOT covered by this
    atomicity at all -- exactly the gap :func:`save_generation` closes."""
    import torch

    os.makedirs(directory, exist_ok=True)
    pt_path = os.path.join(directory, f"{filename}.pt")
    json_path = os.path.join(directory, f"{filename}.json")
    pt_tmp = pt_path + ".tmp"
    json_tmp = json_path + ".tmp"

    payload = {name: _state_dict_or_none(comp) for name, comp in components.items()}
    torch.save(payload, pt_tmp)
    os.replace(pt_tmp, pt_path)

    manifest = dict(meta)
    manifest["present_components"] = sorted(name for name, comp in components.items() if comp is not None)
    with open(json_tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(json_tmp, json_path)


def load_legacy_flat(directory: str, filename: str, components: Dict[str, Any], map_location=None,
                      *, allow_missing: Iterable[str] = ()) -> Dict[str, Any]:
    """Loads a LEGACY flat-file checkpoint (``<filename>.pt``/``<filename>.json``
    directly under ``directory``, no generation/hash cross-verification --
    that machinery didn't exist when this layout was written). Component-
    completeness enforcement (``allow_missing``) is identical to
    :func:`load_generation`. A caller must name this function explicitly;
    there is no auto-detection/fallback between the two layouts."""
    import torch

    pt_path = os.path.join(directory, f"{filename}.pt")
    json_path = os.path.join(directory, f"{filename}.json")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(pt_path)

    payload = torch.load(pt_path, map_location=map_location, weights_only=True)
    manifest = {}
    if os.path.isfile(json_path):
        with open(json_path) as f:
            manifest = json.load(f)

    loaded, skipped, missing_required = _resolve_missing(components, payload, allow_missing)
    _raise_if_missing_required(pt_path, missing_required)
    for name in loaded:
        components[name].load_state_dict(payload[name])

    return {"manifest": manifest, "loaded": sorted(loaded), "skipped": sorted(skipped)}
