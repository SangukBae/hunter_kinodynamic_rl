import os
import shutil
import time

import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.config.schema import RiskConfig, TQCHyperparameters  # noqa: E402
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent as RiskAgent  # noqa: E402
from hunter_kinodynamic_rl.rl.checkpointing import manager  # noqa: E402
from hunter_kinodynamic_rl.rl.replay.buffer import ReplayBuffer  # noqa: E402


def _make_hp() -> TQCHyperparameters:
    return TQCHyperparameters(n_critics=2, n_quantiles=25, top_quantiles_to_drop_per_net=2,
                               batch_size=8, buffer_size=100)


def _make_batch(batch_size, state_dim, action_dim):
    return {
        "state": torch.randn(batch_size, state_dim),
        "action": torch.randn(batch_size, action_dim).clamp(-1, 1),
        "next_state": torch.randn(batch_size, state_dim),
        "reward": torch.randn(batch_size, 1),
        "not_done": torch.ones(batch_size, 1),
        "risk_target": torch.rand(batch_size, 1),
        "valid": torch.ones(batch_size, 1),
    }


def _make_replay(state_dim=4, action_dim=2) -> ReplayBuffer:
    buf = ReplayBuffer(state_dim, action_dim, capacity=100, seed=0, max_candidates=1)
    buf.add(
        state=[0.0] * state_dim, action=[0.0] * action_dim, next_state=[0.0] * state_dim,
        reward=0.0, done=False,
    )
    return buf


# ------------------------------------------------------- legacy flat layout
def test_checkpoint_roundtrip_restores_weights(tmp_path):
    torch.manual_seed(0)
    agent = RiskAgent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=True, actor_lambda=0.1, actor_penalty_warmup_updates=0,
                                               min_valid_labels_per_batch=1),
                       device="cpu")

    # Diverge from init so a "load" that silently no-ops would be caught.
    for _ in range(3):
        agent.train_step(_make_batch(8, 6, 3))

    manager.save_legacy_flat(str(tmp_path), "ckpt", agent.checkpoint_components(),
                              meta={"training_steps": agent.training_steps, "seed": 0,
                                    "ent_coef_state": agent.ent_coef_state()})

    torch.manual_seed(999)  # different seed -> different fresh init
    fresh = RiskAgent(state_dim=6, action_dim=3, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=True, actor_lambda=0.1), device="cpu")
    for p1, p2 in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert not torch.equal(p1, p2), "test setup invalid: fresh agent accidentally matches trained one"

    result = manager.load_legacy_flat(str(tmp_path), "ckpt", fresh.checkpoint_components())
    fresh.load_ent_coef_state(result["manifest"].get("ent_coef_state"))

    assert "actor" in result["loaded"]
    assert "risk_critic" in result["loaded"]
    for p1, p2 in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert torch.equal(p1, p2)
    for p1, p2 in zip(agent.risk_critic.parameters(), fresh.risk_critic.parameters()):
        assert torch.equal(p1, p2)
    assert fresh.ent_coef_state() == pytest.approx(agent.ent_coef_state())


def test_checkpoint_load_reports_skipped_component_when_absent(tmp_path):
    """Loading a risk-disabled checkpoint into a risk-ENABLED agent must not
    crash PROVIDED the caller explicitly opts in via allow_missing (section
    P1-4: a deliberate warm-start, not a silent default) -- the risk critic
    is reported skipped, everything else loads."""
    torch.manual_seed(1)
    plain = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=False), device="cpu")
    manager.save_legacy_flat(str(tmp_path), "ckpt", plain.checkpoint_components(), meta={"training_steps": 0})

    torch.manual_seed(2)
    risky = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=True), device="cpu")
    result = manager.load_legacy_flat(str(tmp_path), "ckpt", risky.checkpoint_components(),
                                       allow_missing={"risk_critic", "risk_critic_optimizer"})
    assert "risk_critic" in result["skipped"]
    assert "actor" in result["loaded"]


def test_checkpoint_load_raises_by_default_on_a_missing_required_component(tmp_path):
    """section P1-4: the core regression -- WITHOUT an explicit
    allow_missing, a missing component the current agent architecture
    actually declares must be a hard failure, never a silent skip."""
    torch.manual_seed(1)
    plain = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=False), device="cpu")
    manager.save_legacy_flat(str(tmp_path), "ckpt", plain.checkpoint_components(), meta={"training_steps": 0})

    torch.manual_seed(2)
    risky = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=True), device="cpu")
    with pytest.raises(RuntimeError, match="risk_critic"):
        manager.load_legacy_flat(str(tmp_path), "ckpt", risky.checkpoint_components())


def test_legacy_load_raises_on_a_truncated_pt_file(tmp_path):
    """A corrupted/incomplete checkpoint (e.g. a process killed mid-copy, a
    disk full during a NON-atomic write, or a manually truncated file) must
    FAIL LOUDLY, never silently load a partial/garbage state -- the atomic
    tmp-then-os.replace save pattern (manager.save_legacy_flat) already
    prevents a genuinely half-WRITTEN file from ever appearing at the real
    checkpoint path, but any .pt file that becomes corrupted or truncated
    AFTER a successful save (disk corruption, an out-of-band edit, a bad
    copy between machines) must still be rejected on load, not silently
    misread."""
    torch.manual_seed(0)
    agent = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=False), device="cpu")
    manager.save_legacy_flat(str(tmp_path), "ckpt", agent.checkpoint_components(), meta={"training_steps": 0})

    pt_path = tmp_path / "ckpt.pt"
    original = pt_path.read_bytes()
    assert len(original) > 100, "test setup sanity: checkpoint file suspiciously small"
    pt_path.write_bytes(original[: len(original) // 2])  # truncate to half

    fresh = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=False), device="cpu")
    with pytest.raises(Exception):
        manager.load_legacy_flat(str(tmp_path), "ckpt", fresh.checkpoint_components())


def test_legacy_load_raises_on_a_missing_checkpoint_file(tmp_path):
    agent = RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                       risk_config=RiskConfig(enabled=False), device="cpu")
    with pytest.raises(FileNotFoundError):
        manager.load_legacy_flat(str(tmp_path), "does_not_exist", agent.checkpoint_components())


# ------------------------------------------------------- generation layout (item-3)
def _make_agent(seed=0, enabled=False):
    torch.manual_seed(seed)
    return RiskAgent(state_dim=4, action_dim=2, max_action=1.0, hyperparameters=_make_hp(),
                      risk_config=RiskConfig(enabled=enabled), device="cpu")


def test_generation_roundtrip_restores_weights_and_replay(tmp_path):
    agent = _make_agent(seed=0)
    for _ in range(2):
        agent.train_step(_make_batch(8, 4, 2))
    replay = _make_replay()

    generation = manager.save_generation(
        str(tmp_path), "ckpt", agent.checkpoint_components(),
        meta={"training_steps": agent.training_steps}, replay_buffer=replay,
    )
    assert generation  # a real, non-empty id was generated
    ckpt_path = tmp_path / "ckpt"
    assert os.path.islink(str(ckpt_path))
    assert os.path.isdir(str(ckpt_path))  # symlink resolves to a real directory
    assert (ckpt_path / "model.pt").is_file()
    assert (ckpt_path / "manifest.json").is_file()
    assert (ckpt_path / "replay.npz").is_file()

    fresh = _make_agent(seed=999)
    for p1, p2 in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert not torch.equal(p1, p2), "test setup invalid"

    result = manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())
    assert result["generation"] == generation
    assert "actor" in result["loaded"]
    for p1, p2 in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert torch.equal(p1, p2)

    restored_replay = ReplayBuffer.load(result["replay_path"], seed=0)
    assert restored_replay.size == replay.size
    assert restored_replay.generation == generation


def test_generation_load_raises_on_missing_component_by_default(tmp_path):
    plain = _make_agent(seed=1, enabled=False)
    manager.save_generation(str(tmp_path), "ckpt", plain.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())

    risky = _make_agent(seed=2, enabled=True)
    with pytest.raises(RuntimeError, match="risk_critic"):
        manager.load_generation(str(tmp_path), "ckpt", risky.checkpoint_components())

    result = manager.load_generation(str(tmp_path), "ckpt", risky.checkpoint_components(),
                                      allow_missing={"risk_critic", "risk_critic_optimizer"})
    assert "risk_critic" in result["skipped"]


def test_generation_load_raises_on_missing_checkpoint(tmp_path):
    agent = _make_agent()
    with pytest.raises(FileNotFoundError):
        manager.load_generation(str(tmp_path), "does_not_exist", agent.checkpoint_components())


def test_generation_load_raises_on_missing_replay_file(tmp_path):
    """section item-3: an incomplete generation (model+manifest written,
    replay somehow absent) must fail loudly, never silently resume with an
    empty replay buffer."""
    agent = _make_agent()
    manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())
    generation_dirs = list((tmp_path / ".generations").iterdir())
    assert len(generation_dirs) == 1
    os.remove(str(generation_dirs[0] / "replay.npz"))

    fresh = _make_agent()
    with pytest.raises(FileNotFoundError, match="replay.npz"):
        manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())


def test_generation_load_raises_on_damaged_model_pt(tmp_path):
    agent = _make_agent()
    manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())
    generation_dir = next((tmp_path / ".generations").iterdir())
    pt_path = generation_dir / "model.pt"
    original = pt_path.read_bytes()
    pt_path.write_bytes(original[: len(original) // 2])

    fresh = _make_agent()
    with pytest.raises(Exception):
        manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())


def test_generation_load_raises_when_model_pt_is_swapped_for_a_different_generation(tmp_path):
    """section item-3: the CORE new regression -- swap ONLY the .pt file for
    a DIFFERENT (but individually well-formed, undamaged) generation's own
    model.pt. Must be detected via the embedded __generation__/hash
    cross-check, never silently loaded as if nothing happened."""
    agent_a = _make_agent(seed=1)
    generation_a = manager.save_generation(str(tmp_path), "ckpt_a", agent_a.checkpoint_components(),
                                            meta={}, replay_buffer=_make_replay())
    agent_b = _make_agent(seed=2)
    generation_b = manager.save_generation(str(tmp_path), "ckpt_b", agent_b.checkpoint_components(),
                                            meta={}, replay_buffer=_make_replay())
    assert generation_a != generation_b

    # Swap ckpt_a's model.pt for ckpt_b's own (both are individually
    # well-formed, undamaged files -- only the PAIRING is wrong).
    ckpt_a_dir = os.path.realpath(str(tmp_path / "ckpt_a"))
    ckpt_b_dir = os.path.realpath(str(tmp_path / "ckpt_b"))
    shutil.copyfile(os.path.join(ckpt_b_dir, "model.pt"), os.path.join(ckpt_a_dir, "model.pt"))

    fresh = _make_agent()
    with pytest.raises(RuntimeError, match="generation mismatch"):
        manager.load_generation(str(tmp_path), "ckpt_a", fresh.checkpoint_components())


def test_generation_load_raises_when_replay_npz_is_swapped_for_a_different_generation(tmp_path):
    """The dual of the model.pt-swap regression above -- swapping ONLY
    replay.npz for a different generation's own must also be caught."""
    agent_a = _make_agent(seed=1)
    manager.save_generation(str(tmp_path), "ckpt_a", agent_a.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())
    agent_b = _make_agent(seed=2)
    manager.save_generation(str(tmp_path), "ckpt_b", agent_b.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())

    ckpt_a_dir = os.path.realpath(str(tmp_path / "ckpt_a"))
    ckpt_b_dir = os.path.realpath(str(tmp_path / "ckpt_b"))
    shutil.copyfile(os.path.join(ckpt_b_dir, "replay.npz"), os.path.join(ckpt_a_dir, "replay.npz"))

    fresh = _make_agent()
    with pytest.raises(RuntimeError, match="generation mismatch"):
        manager.load_generation(str(tmp_path), "ckpt_a", fresh.checkpoint_components())


def test_generation_load_raises_when_manifest_generation_is_hand_edited(tmp_path):
    """A manifest.json whose OWN recorded `generation` field was
    hand-edited (e.g. to paper over a mismatch) must still be caught -- the
    embedded model.pt/replay.npz generations would then disagree with the
    (tampered) manifest."""
    import json

    agent = _make_agent()
    manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())
    generation_dir = next((tmp_path / ".generations").iterdir())
    manifest_path = generation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generation"] = "hand-edited-not-the-real-one"
    manifest_path.write_text(json.dumps(manifest))

    fresh = _make_agent()
    with pytest.raises(RuntimeError, match="generation mismatch"):
        manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())


def test_generation_load_raises_when_replay_content_edited_without_updating_hash(tmp_path):
    """A replay.npz that keeps the SAME generation tag but whose actual
    bytes were modified out-of-band (so its recorded size/sha256 in
    manifest.json no longer match) must be rejected, distinct from the
    generation-tag check above -- only the size/hash comparisons can catch
    it. Since item-5 (round 3) added an independent SIZE check ahead of
    the sha256 one, whichever of the two the edited content happens to
    trip first is an acceptable outcome here; both are exercised
    explicitly by their own dedicated tests
    (test_load_generation_raises_on_manifest_*_size_mismatch_even_with_matching_sha256)."""
    agent = _make_agent()
    replay = _make_replay()
    generation = manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                                          meta={}, replay_buffer=replay)
    generation_dir = tmp_path / ".generations" / generation
    replay_path = generation_dir / "replay.npz"
    # Re-save a DIFFERENT replay buffer's content directly at the same path
    # WITH the same generation tag (so the generation check alone wouldn't
    # catch it) -- only the size/hash comparisons can.
    other_replay = _make_replay()
    other_replay.add(state=[1.0] * 4, action=[1.0] * 2, next_state=[1.0] * 4, reward=1.0, done=False)
    other_replay.save(str(replay_path), generation=generation)

    fresh = _make_agent()
    with pytest.raises(RuntimeError, match="checkpoint set inconsistency"):
        manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())


def test_generation_layout_is_not_readable_via_load_legacy_flat(tmp_path):
    """section item-3: the two layouts are never silently interchangeable
    -- a generation-layout checkpoint (a directory) must not be
    misinterpreted by load_legacy_flat (which expects a flat `<tag>.pt`
    file) as if it were simply missing."""
    agent = _make_agent()
    manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())
    with pytest.raises(FileNotFoundError):
        manager.load_legacy_flat(str(tmp_path), "ckpt", agent.checkpoint_components())


# --------------------------------------------------- item-3 (round 2): TOCTOU fix
def test_load_generation_replay_path_stays_pinned_to_the_verified_generation_even_if_tag_is_later_repointed(
        tmp_path):
    """The CORE round-2 regression, reproducing exactly the race named in
    the governing instruction: verify against generation A, then have the
    SAME tag repointed to a DIFFERENT generation B, then load the replay
    buffer from the path load_generation returned -- it must still be A's,
    never silently B's. Before the fix, `result["replay_path"]` was
    `directory/tag/replay.npz` -- a path that re-traverses the `tag`
    symlink on every fresh open, so it would have picked up B."""
    agent_a = _make_agent(seed=1)
    replay_a = _make_replay()
    generation_a = manager.save_generation(str(tmp_path), "ckpt", agent_a.checkpoint_components(),
                                            meta={}, replay_buffer=replay_a)

    fresh = _make_agent()
    result = manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())
    assert result["generation"] == generation_a

    # Simulate a concurrent save republishing the SAME tag to a DIFFERENT
    # generation AFTER this load's own verification completed but BEFORE
    # the caller gets around to loading the replay buffer from the path
    # handed back -- the exact race item-3 (round 2) describes.
    agent_b = _make_agent(seed=2)
    replay_b = _make_replay()
    replay_b.add(state=[9.0] * 4, action=[9.0] * 2, next_state=[9.0] * 4, reward=9.0, done=False)
    generation_b = manager.save_generation(str(tmp_path), "ckpt", agent_b.checkpoint_components(),
                                            meta={}, replay_buffer=replay_b)
    assert generation_b != generation_a

    # The FIX: result["replay_path"] must still resolve to generation A's
    # OWN replay.npz -- never silently re-resolve through the (now
    # repointed) "ckpt" symlink to generation B's.
    restored_replay = ReplayBuffer.load(result["replay_path"], seed=0)
    assert restored_replay.generation == generation_a
    assert restored_replay.size == replay_a.size  # NOT replay_b's (which has +1 entry)
    assert result["generation_dir"] == os.path.realpath(os.path.join(str(tmp_path), ".generations", generation_a))


def test_load_generation_returns_the_resolved_generation_dir(tmp_path):
    agent = _make_agent()
    generation = manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                                          meta={}, replay_buffer=_make_replay())
    result = manager.load_generation(str(tmp_path), "ckpt", agent.checkpoint_components())
    assert result["generation_dir"] == os.path.join(str(tmp_path), ".generations", generation)
    assert os.path.isdir(result["generation_dir"])


def test_concurrent_save_and_load_stress_never_mixes_generations(tmp_path):
    """A REAL (not mocked) concurrency stress test: multiple threads
    repeatedly calling save_generation against the SAME tag while OTHER
    threads repeatedly load_generation + immediately load the returned
    replay_path, asserting the replay's own embedded generation always
    matches the manifest's -- proving the TOCTOU fix holds under genuine
    concurrent file-system access, not just the single deterministic
    interleaving reproduced above."""
    import threading

    directory = str(tmp_path)
    errors = []

    # Seed one generation first so loaders always have something to find.
    manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())

    def _saver(seed_base: int) -> None:
        for i in range(8):
            try:
                agent = _make_agent(seed=seed_base + i)
                manager.save_generation(directory, "ckpt", agent.checkpoint_components(),
                                         meta={}, replay_buffer=_make_replay())
            except Exception as e:  # noqa: BLE001 -- collected and asserted on below
                errors.append(e)

    def _loader() -> None:
        for _ in range(15):
            try:
                fresh = _make_agent()
                result = manager.load_generation(directory, "ckpt", fresh.checkpoint_components())
                restored = ReplayBuffer.load(result["replay_path"], seed=0)
                if restored.generation != result["generation"]:
                    errors.append(AssertionError(
                        f"mismatched generation: replay={restored.generation!r} "
                        f"manifest={result['generation']!r}"
                    ))
            except FileNotFoundError:
                pass  # a save was mid-flight when this load started -- not a bug, just a timing gap

    threads = (
        [threading.Thread(target=_saver, args=(100 * i,)) for i in range(2)]
        + [threading.Thread(target=_loader) for _ in range(4)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors


def test_publish_generation_symlink_survives_a_concurrent_publish_to_the_same_tag(tmp_path):
    """Regression for a REAL bug the stress test above found: the staging
    symlink path used to publish a generation was a FIXED name
    (``.{tag}.symlink.tmp``) shared by every save to the same tag -- two
    concurrent save_generation() calls could both pass the
    ``os.path.lexists`` check before either created the file, then both
    call ``os.symlink`` on the identical path, raising FileExistsError on
    the second one. A per-generation-unique staging name (this fix) makes
    that collision structurally impossible: directly exercised here by
    calling the internal publish step twice, back-to-back, for two
    DIFFERENT generations at the SAME tag with no directory cleanup
    between them (the scenario that used to intermittently fail)."""
    directory = str(tmp_path)
    os.makedirs(os.path.join(directory, ".generations", "gen-a"))
    os.makedirs(os.path.join(directory, ".generations", "gen-b"))

    manager._publish_generation_symlink(directory, "ckpt", "gen-a")
    manager._publish_generation_symlink(directory, "ckpt", "gen-b")  # must not raise FileExistsError

    assert os.path.realpath(os.path.join(directory, "ckpt")) == os.path.realpath(
        os.path.join(directory, ".generations", "gen-b"))


# --------------------------------------------------- item-5 (round 3): size verification
def test_load_generation_raises_on_manifest_pt_size_mismatch_even_with_matching_sha256(tmp_path):
    """The core item-5 (round 3) regression: manifest.json's `pt_size_bytes`
    field was recorded at save time and this module's OWN docstring
    claimed it was verified at load, but the code never actually checked
    it -- only the sha256. Proven here by corrupting ONLY the manifest's
    recorded size (never model.pt itself, so its sha256 still matches the
    manifest's own recorded pt_sha256) -- the size check must catch this
    independently."""
    import json

    agent = _make_agent()
    generation = manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                                          meta={}, replay_buffer=_make_replay())
    manifest_path = os.path.join(str(tmp_path), ".generations", generation, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["pt_size_bytes"] = manifest["pt_size_bytes"] + 1000  # tampered; model.pt itself untouched
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    fresh = _make_agent()
    with pytest.raises(RuntimeError, match="pt_size_bytes"):
        manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())


def test_load_generation_raises_on_manifest_replay_size_mismatch_even_with_matching_sha256(tmp_path):
    import json

    agent = _make_agent()
    generation = manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                                          meta={}, replay_buffer=_make_replay())
    manifest_path = os.path.join(str(tmp_path), ".generations", generation, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["replay_size_bytes"] = manifest["replay_size_bytes"] + 1000  # tampered; replay.npz untouched
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    fresh = _make_agent()
    with pytest.raises(RuntimeError, match="replay_size_bytes"):
        manager.load_generation(str(tmp_path), "ckpt", fresh.checkpoint_components())


# --------------------------------------------------------- item-5 (round 3): parent-dir fsync
def test_save_generation_fsyncs_the_generations_parent_directory(tmp_path, monkeypatch):
    """Can't directly observe an fsync's effect in a unit test (no crash
    to survive), but can confirm save_generation actually CALLS
    _fsync_path against the `.generations` parent directory, not just the
    generation's own subdirectory -- the concrete, code-level fix the
    governing instruction asked for."""
    fsynced_paths = []
    real_fsync_path = manager._fsync_path
    monkeypatch.setattr(manager, "_fsync_path", lambda path: (fsynced_paths.append(path), real_fsync_path(path))[1])

    agent = _make_agent()
    manager.save_generation(str(tmp_path), "ckpt", agent.checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())

    generations_dir = os.path.join(str(tmp_path), ".generations")
    assert generations_dir in fsynced_paths


# --------------------------------------------------- item-3 (round 2): orphan cleanup
def test_prune_orphan_generations_removes_only_old_unreferenced_ones(tmp_path, monkeypatch):
    """mark-then-sweep (item-5 round 3): the FIRST prune call only MARKS
    gen_a as orphaned (nothing is old enough to delete yet, regardless of
    min_age_sec) -- only a LATER call, once min_age_sec has elapsed since
    THAT marker was written, actually deletes it."""
    import time as time_mod

    directory = str(tmp_path)
    gen_a = manager.save_generation(directory, "ckpt", _make_agent(seed=1).checkpoint_components(),
                                     meta={}, replay_buffer=_make_replay())
    gen_b = manager.save_generation(directory, "ckpt", _make_agent(seed=2).checkpoint_components(),
                                     meta={}, replay_buffer=_make_replay())  # currently referenced by "ckpt"

    first_pass = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    assert first_pass == []  # marks gen_a as orphaned, deletes nothing yet
    assert os.path.isdir(os.path.join(directory, ".generations", gen_a))

    # Advance simulated time past min_age_sec SINCE THE MARKER was written.
    real_time = time_mod.time
    monkeypatch.setattr(time_mod, "time", lambda: real_time() + 7200.0)

    removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    assert removed == [gen_a]
    assert not os.path.isdir(os.path.join(directory, ".generations", gen_a))
    # gen_b is still referenced (the live "ckpt" tag) -- must survive.
    assert os.path.isdir(os.path.join(directory, ".generations", gen_b))
    result = manager.load_generation(directory, "ckpt", _make_agent().checkpoint_components())
    assert result["generation"] == gen_b


def test_prune_orphan_generations_never_deletes_a_freshly_orphaned_generation_even_if_it_is_old(tmp_path):
    """The CORE item-5 (round 3) regression this whole mark-then-sweep
    redesign exists for: a generation that was REFERENCED continuously for
    a long time (an ANCIENT on-disk creation mtime -- e.g. a long-lived
    checkpoint resumed for hours) and only became orphaned a MOMENT ago
    must NOT be deleted on the very next prune call. The pre-fix,
    mtime-based aging would have incorrectly treated it as already
    eligible purely because it happened to be old, exactly the concurrent-
    delete race a reader that just resolved this same tag could be caught
    in."""
    directory = str(tmp_path)
    gen = manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                                   meta={}, replay_buffer=_make_replay())
    gen_dir = os.path.join(directory, ".generations", gen)
    ancient = time.time() - 999999.0
    os.utime(gen_dir, (ancient, ancient))  # simulate a long-lived, ancient generation
    os.remove(os.path.join(directory, "ckpt"))  # ONLY JUST orphaned now

    removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    assert removed == []  # not eligible -- only just observed orphaned THIS call
    assert os.path.isdir(gen_dir)


def test_prune_orphan_generations_clears_the_marker_if_referenced_again(tmp_path):
    """A generation that flips back to referenced between two prune calls
    (e.g. a manual rollback re-publishing an older generation) must have
    its orphan marker cleared -- a LATER re-orphaning must restart its own
    min_age_sec window from scratch, never inherit a stale marker from a
    previous orphan period."""
    directory = str(tmp_path)
    gen = manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                                   meta={}, replay_buffer=_make_replay())
    os.remove(os.path.join(directory, "ckpt"))
    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # marks it
    marker_path = os.path.join(directory, ".generations", gen, manager._ORPHAN_MARKER_FILENAME)
    assert os.path.isfile(marker_path)

    manager._publish_generation_symlink(directory, "ckpt", gen)  # referenced again (e.g. a rollback)
    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    assert not os.path.isfile(marker_path)


def test_prune_orphan_generations_respects_keep_last_n(tmp_path, monkeypatch):
    import time as time_mod

    directory = str(tmp_path)
    orphans = []
    real_time = time_mod.time
    for i in range(4):
        gen = manager.save_generation(directory, f"ckpt_{i}", _make_agent(seed=i).checkpoint_components(),
                                       meta={}, replay_buffer=_make_replay())
        orphans.append(gen)
        os.remove(os.path.join(directory, f"ckpt_{i}"))  # unpublish -- now an orphan
        # Mark it as orphaned NOW, at a distinct, monotonically increasing
        # simulated time per generation, so keep_last_n has an unambiguous
        # "most recently orphaned" ordering to select from.
        monkeypatch.setattr(time_mod, "time", lambda offset=float(i): real_time() + offset)
        manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # marks only

    # Advance far enough that every marker is now past min_age_sec.
    monkeypatch.setattr(time_mod, "time", lambda: real_time() + 7200.0)
    removed = manager.prune_orphan_generations(directory, keep_last_n=1, min_age_sec=3600.0)
    assert len(removed) == 3  # kept the 1 most-recently-orphaned
    assert orphans[-1] not in removed  # the most recently orphaned one survives
    surviving = os.listdir(os.path.join(directory, ".generations"))
    assert surviving == [orphans[-1]]


def test_prune_orphan_generations_never_removes_anything_younger_than_min_age_since_marking(tmp_path):
    directory = str(tmp_path)
    gen = manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                                   meta={}, replay_buffer=_make_replay())
    os.remove(os.path.join(directory, "ckpt"))  # unpublish -- now an orphan, but BRAND NEW

    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # marks it
    removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # too soon since marking
    assert removed == []
    assert os.path.isdir(os.path.join(directory, ".generations", gen))


def test_prune_orphan_generations_on_a_directory_with_no_generations_is_a_noop(tmp_path):
    assert manager.prune_orphan_generations(str(tmp_path)) == []


# --------------------------------------------------- item-3 (checkpoint-prune-safety fix)
def _make_deletable_orphan(tmp_path, monkeypatch, min_age_sec=3600.0):
    """Produces one generation that is orphaned, marked, and aged past
    min_age_sec -- i.e. genuinely ELIGIBLE for deletion on the next prune
    call -- for tests that exercise what happens right at the point of
    deletion."""
    import time as time_mod

    directory = str(tmp_path)
    gen = manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                                   meta={}, replay_buffer=_make_replay())
    os.remove(os.path.join(directory, "ckpt"))  # orphan it
    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=min_age_sec)  # marks it

    real_time = time_mod.time
    monkeypatch.setattr(time_mod, "time", lambda: real_time() + min_age_sec + 1.0)
    return directory, gen


def test_prune_already_running_raises_and_deletes_nothing(tmp_path, monkeypatch):
    """The core 'enforce non-concurrent execution in code' regression: a
    SECOND prune call against the SAME directory while a first one already
    holds the directory-level exclusive lock must be rejected outright,
    never silently interleave with it."""
    directory, gen = _make_deletable_orphan(tmp_path, monkeypatch)

    lock_path = os.path.join(directory, manager._PRUNE_LOCK_FILENAME)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    import fcntl
    fcntl.flock(lock_fd, fcntl.LOCK_EX)  # simulates a first prune call already in progress
    try:
        with pytest.raises(manager.PruneAlreadyRunningError):
            manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    # Nothing was deleted -- the second call bailed out before touching anything.
    assert os.path.isdir(os.path.join(directory, ".generations", gen))


def test_prune_skips_a_generation_currently_locked_by_a_concurrent_load(tmp_path, monkeypatch):
    """The core TOCTOU-fix regression this whole mechanism exists for: a
    generation that is otherwise fully eligible for deletion (orphaned,
    marked, aged past min_age_sec, unreferenced) must NOT be deleted while
    something else holds a lock on it (standing in for an in-flight
    load_generation() read) -- proving the fix is a REAL lock acquisition,
    not just the pre-existing (and, on its own, insufficient) "recheck
    referenced" pass."""
    directory, gen = _make_deletable_orphan(tmp_path, monkeypatch)
    gen_dir = os.path.join(directory, ".generations", gen)

    lock_path = os.path.join(gen_dir, manager._GENERATION_LOCK_FILENAME)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    import fcntl
    fcntl.flock(lock_fd, fcntl.LOCK_SH)  # simulates a concurrent load_generation() read in progress
    try:
        removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    assert removed == []  # skipped, not deleted, while the lock was held
    assert os.path.isdir(gen_dir)


def test_load_generation_holds_a_lock_that_a_concurrent_prune_cannot_override(tmp_path, monkeypatch):
    """The other half of the same guarantee, exercised through the REAL
    public API instead of manually poking the lock file: a genuine
    load_generation() call, artificially slowed down mid-read, must keep a
    concurrently-running prune from deleting the generation it's reading."""
    import threading
    import time as time_mod

    directory, gen = _make_deletable_orphan(tmp_path, monkeypatch)
    gen_dir = os.path.join(directory, ".generations", gen)
    # Re-publish "ckpt" at this same generation so load_generation can find
    # it by tag (the helper above orphaned it on purpose to make it
    # prune-eligible) -- loading by tag, not by directly touching gen_dir,
    # is what real callers do.
    manager._publish_generation_symlink(directory, "ckpt", gen)

    load_started = threading.Event()
    release_load = threading.Event()
    real_read = manager._read_and_verify_generation

    def _slow_read(*args, **kwargs):
        load_started.set()
        release_load.wait(timeout=5.0)
        return real_read(*args, **kwargs)

    monkeypatch.setattr(manager, "_read_and_verify_generation", _slow_read)

    result_holder = {}

    def _do_load():
        fresh = _make_agent()
        result_holder["result"] = manager.load_generation(directory, "ckpt", fresh.checkpoint_components())

    t = threading.Thread(target=_do_load)
    t.start()
    assert load_started.wait(timeout=5.0)

    # "ckpt" is still published (never unpublished here), so a real prune
    # wouldn't even consider gen orphaned -- unpublish it now, exactly the
    # window load_generation's own resolve-then-read gap is vulnerable to,
    # then let prune's SECOND mark-then-sweep pass (this test's
    # _make_deletable_orphan already advanced simulated time far enough)
    # see it as a fully-eligible orphan.
    os.remove(os.path.join(directory, "ckpt"))
    removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    assert removed == []  # the in-flight load's shared lock blocked the delete
    assert os.path.isdir(gen_dir)

    release_load.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert result_holder["result"]["generation"] == gen


def test_save_generation_lifecycle_excludes_prune_and_marker_until_publish(tmp_path, monkeypatch):
    """Counterexample for the save-side gap: a fresh generation directory is
    unreferenced by any tag until _publish_generation_symlink runs, so a
    concurrent prune (given a small enough min_age_sec) could otherwise
    mark it orphaned and, on a later call once that marker ages out, delete
    it out from under an in-flight save -- BEFORE the fix, this test's
    second prune call actually deletes gen_dir while the save thread is
    paused right before publish; after the fix, save_generation holds the
    generation's own exclusive lock across that whole window, so prune's
    non-blocking attempt on the same lock file must skip it."""
    import threading

    directory = str(tmp_path)
    publish_started = threading.Event()
    release_publish = threading.Event()
    real_publish = manager._publish_generation_symlink

    def _slow_publish(*args, **kwargs):
        publish_started.set()
        release_publish.wait(timeout=5.0)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(manager, "_publish_generation_symlink", _slow_publish)

    result_holder = {}

    def _do_save():
        result_holder["gen"] = manager.save_generation(
            directory, "ckpt", _make_agent().checkpoint_components(), meta={}, replay_buffer=_make_replay())

    t = threading.Thread(target=_do_save)
    t.start()
    try:
        assert publish_started.wait(timeout=5.0)

        gens = os.listdir(os.path.join(directory, ".generations"))
        assert len(gens) == 1
        gen = gens[0]
        gen_dir = os.path.join(directory, ".generations", gen)

        # Publish has not happened yet, but the save's lifecycle lock makes
        # this directory ineligible even for orphan marking.
        assert manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=0.0) == []
        marker = os.path.join(gen_dir, manager._ORPHAN_MARKER_FILENAME)
        assert not os.path.exists(marker)  # lifecycle lock: in-flight saves are not orphan candidates at all
        import time as time_mod
        real_time = time_mod.time
        monkeypatch.setattr(time_mod, "time", lambda: real_time() + 1.0)
        # Second pass: marker is now "old enough" -- fully eligible by every
        # criterion EXCEPT the in-flight save's own lock.
        removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=0.0)
        assert removed == []  # the fix: save_generation's own lock blocked the delete
        assert os.path.isdir(gen_dir)
    finally:
        release_publish.set()
        t.join(timeout=5.0)
    assert not t.is_alive()
    assert result_holder["gen"] == gen
    assert os.path.realpath(os.path.join(directory, "ckpt")) == gen_dir
    assert not os.path.exists(marker)  # no stale pre-publication age can leak into a future orphan period


def test_load_generation_lease_blocks_a_concurrent_prune_during_replay_deserialization(tmp_path, monkeypatch):
    """The TOCTOU counterexample this section responds to: load_generation()
    alone releases its shared lock the instant it returns, BEFORE a caller
    like trainer_base._resume_from goes on to call
    ReplayBuffer.load(result["replay_path"]) -- so a concurrent prune could
    delete the generation in between. load_generation_lease must keep the
    SAME lock held through that whole window; this test drives a real
    ReplayBuffer.load (artificially slowed) inside the lease's `with` block
    and proves a concurrent prune attempted mid-deserialization is skipped,
    not raced."""
    import threading

    directory, gen = _make_deletable_orphan(tmp_path, monkeypatch)
    gen_dir = os.path.join(directory, ".generations", gen)
    manager._publish_generation_symlink(directory, "ckpt", gen)

    replay_load_started = threading.Event()
    release_replay_load = threading.Event()
    real_replay_load = ReplayBuffer.load.__func__

    def _slow_replay_load(cls, path, *args, **kwargs):
        replay_load_started.set()
        release_replay_load.wait(timeout=5.0)
        return real_replay_load(cls, path, *args, **kwargs)

    monkeypatch.setattr(ReplayBuffer, "load", classmethod(_slow_replay_load))

    result_holder = {}

    def _do_resume():
        with manager.load_generation_lease(directory, "ckpt", _make_agent().checkpoint_components()) as result:
            result_holder["replay"] = ReplayBuffer.load(result["replay_path"], seed=0)

    t = threading.Thread(target=_do_resume)
    t.start()
    try:
        assert replay_load_started.wait(timeout=5.0)

        # Same window a real concurrent save republishing "ckpt" would open:
        # this generation just became unreferenced while still being read.
        os.remove(os.path.join(directory, "ckpt"))
        removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
        assert removed == []  # the lease's shared lock blocked the delete
        assert os.path.isdir(gen_dir)
    finally:
        release_replay_load.set()
        t.join(timeout=5.0)
    assert not t.is_alive()
    assert result_holder["replay"] is not None


def test_load_generation_alone_does_not_protect_a_caller_reading_replay_path_after_return(tmp_path, monkeypatch):
    """Documents load_generation()'s intentionally narrow contract: its lock
    is released the instant it returns, so touching result["replay_path"]
    AFTER that call -- rather than inside load_generation_lease's `with`
    block -- gets no protection at all: a prune that becomes eligible in
    that window can and does delete the generation first."""
    directory, gen = _make_deletable_orphan(tmp_path, monkeypatch)
    manager._publish_generation_symlink(directory, "ckpt", gen)
    result = manager.load_generation(directory, "ckpt", _make_agent().checkpoint_components())

    os.remove(os.path.join(directory, "ckpt"))
    removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)
    assert removed == [gen]
    with pytest.raises(FileNotFoundError):
        ReplayBuffer.load(result["replay_path"], seed=0)


def test_prune_reports_a_real_rmtree_failure_instead_of_silently_treating_it_as_deleted(tmp_path, monkeypatch):
    """section item-3: shutil.rmtree failures must never be swallowed
    (ignore_errors=True, the pre-fix behavior) and the failed generation
    must never appear in the returned/deleted list."""
    directory, gen = _make_deletable_orphan(tmp_path, monkeypatch)
    gen_dir = os.path.join(directory, ".generations", gen)

    def _raising_rmtree(path, *a, **kw):
        raise PermissionError(f"simulated: cannot remove {path}")

    monkeypatch.setattr(manager.shutil, "rmtree", _raising_rmtree)

    with pytest.raises(manager.PrunePartialFailureError) as exc_info:
        manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)

    assert exc_info.value.deleted == []
    assert gen in exc_info.value.failed
    assert "simulated" in exc_info.value.failed[gen]
    assert os.path.isdir(gen_dir)  # never removed


def test_prune_a_mix_of_a_failing_and_a_succeeding_deletion_returns_only_the_real_successes(tmp_path, monkeypatch):
    """Multiple eligible candidates in one call: one whose rmtree fails,
    one that deletes cleanly -- the exception's .deleted must contain ONLY
    the one that actually succeeded, never the failed one, and the
    succeeding one really is gone on disk."""
    import time as time_mod

    directory = str(tmp_path)
    gen_ok = manager.save_generation(directory, "ckpt_ok", _make_agent(seed=1).checkpoint_components(),
                                      meta={}, replay_buffer=_make_replay())
    gen_bad = manager.save_generation(directory, "ckpt_bad", _make_agent(seed=2).checkpoint_components(),
                                       meta={}, replay_buffer=_make_replay())
    os.remove(os.path.join(directory, "ckpt_ok"))
    os.remove(os.path.join(directory, "ckpt_bad"))
    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # marks both

    real_time = time_mod.time
    monkeypatch.setattr(time_mod, "time", lambda: real_time() + 7200.0)

    real_rmtree = manager.shutil.rmtree

    def _selective_rmtree(path, *a, **kw):
        if gen_bad in path:
            raise PermissionError(f"simulated: cannot remove {path}")
        return real_rmtree(path, *a, **kw)

    monkeypatch.setattr(manager.shutil, "rmtree", _selective_rmtree)

    with pytest.raises(manager.PrunePartialFailureError) as exc_info:
        manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)

    assert exc_info.value.deleted == [gen_ok]
    assert list(exc_info.value.failed) == [gen_bad]
    assert not os.path.isdir(os.path.join(directory, ".generations", gen_ok))
    assert os.path.isdir(os.path.join(directory, ".generations", gen_bad))


def test_orphan_marker_creation_fsyncs_its_own_generation_directory(tmp_path, monkeypatch):
    directory = str(tmp_path)
    gen = manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                                   meta={}, replay_buffer=_make_replay())
    os.remove(os.path.join(directory, "ckpt"))

    fsynced = []
    real_fsync_path = manager._fsync_path
    monkeypatch.setattr(manager, "_fsync_path", lambda path: (fsynced.append(path), real_fsync_path(path))[1])

    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # marks -- creates the marker

    gen_dir = os.path.join(directory, ".generations", gen)
    assert gen_dir in fsynced


def test_orphan_marker_clear_fsyncs_its_own_generation_directory(tmp_path, monkeypatch):
    directory = str(tmp_path)
    gen = manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                                   meta={}, replay_buffer=_make_replay())
    os.remove(os.path.join(directory, "ckpt"))
    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # marks it
    manager._publish_generation_symlink(directory, "ckpt", gen)  # referenced again

    fsynced = []
    real_fsync_path = manager._fsync_path
    monkeypatch.setattr(manager, "_fsync_path", lambda path: (fsynced.append(path), real_fsync_path(path))[1])

    manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)  # clears the marker

    gen_dir = os.path.join(directory, ".generations", gen)
    assert gen_dir in fsynced


def test_prune_fsyncs_the_generations_parent_directory_after_a_real_deletion(tmp_path, monkeypatch):
    directory, gen = _make_deletable_orphan(tmp_path, monkeypatch)

    fsynced = []
    real_fsync_path = manager._fsync_path
    monkeypatch.setattr(manager, "_fsync_path", lambda path: (fsynced.append(path), real_fsync_path(path))[1])

    removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)

    assert removed == [gen]
    generations_dir = os.path.join(directory, ".generations")
    assert generations_dir in fsynced


def test_prune_does_not_fsync_generations_dir_when_nothing_was_actually_deleted(tmp_path, monkeypatch):
    """Only fsync `.generations` when a deletion genuinely happened --
    nothing changed about its directory entries otherwise."""
    directory = str(tmp_path)
    manager.save_generation(directory, "ckpt", _make_agent().checkpoint_components(),
                             meta={}, replay_buffer=_make_replay())  # still referenced -- nothing to prune

    fsynced = []
    real_fsync_path = manager._fsync_path
    monkeypatch.setattr(manager, "_fsync_path", lambda path: (fsynced.append(path), real_fsync_path(path))[1])

    removed = manager.prune_orphan_generations(directory, keep_last_n=0, min_age_sec=3600.0)

    assert removed == []
    generations_dir = os.path.join(directory, ".generations")
    assert generations_dir not in fsynced
