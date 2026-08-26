import pytest

from hunter_kinodynamic_rl.config.schema import ScenarioConfig
from hunter_kinodynamic_rl.env.scenarios.seed_scheduler import SeedPoolViolation, SeedScheduler


def make_cfg() -> ScenarioConfig:
    return ScenarioConfig(train_seed_range=[0, 99], validation_seed_range=[100, 149], test_seed_range=[200, 249])


def test_next_seed_stays_within_mode_range():
    sched = SeedScheduler(run_seed=1, scenario_cfg=make_cfg(), mode="train")
    for _ in range(50):
        s = sched.next_seed()
        assert 0 <= s <= 99


def test_validation_seeds_stay_within_validation_range():
    sched = SeedScheduler(run_seed=1, scenario_cfg=make_cfg(), mode="validation")
    for _ in range(20):
        s = sched.next_seed()
        assert 100 <= s <= 149


def test_same_run_seed_reproduces_same_sequence():
    cfg = make_cfg()
    s1 = SeedScheduler(run_seed=7, scenario_cfg=cfg, mode="train")
    s2 = SeedScheduler(run_seed=7, scenario_cfg=cfg, mode="train")
    seq1 = [s1.next_seed() for _ in range(10)]
    seq2 = [s2.next_seed() for _ in range(10)]
    assert seq1 == seq2


def test_different_run_seed_produces_different_sequence():
    cfg = make_cfg()
    s1 = SeedScheduler(run_seed=7, scenario_cfg=cfg, mode="train")
    s2 = SeedScheduler(run_seed=8, scenario_cfg=cfg, mode="train")
    seq1 = [s1.next_seed() for _ in range(10)]
    seq2 = [s2.next_seed() for _ in range(10)]
    assert seq1 != seq2


def test_episode_index_advances_and_resume_continues_sequence():
    cfg = make_cfg()
    sched = SeedScheduler(run_seed=3, scenario_cfg=cfg, mode="train")
    first_five = [sched.next_seed() for _ in range(5)]
    state = sched.state_dict()
    assert state["episode_index"] == 5

    resumed = SeedScheduler.from_state_dict(state, cfg)
    next_from_resumed = resumed.next_seed()

    fresh = SeedScheduler(run_seed=3, scenario_cfg=cfg, mode="train")
    for _ in range(5):
        fresh.next_seed()
    next_from_fresh = fresh.next_seed()
    assert next_from_resumed == next_from_fresh


def test_validate_explicit_seed_accepts_matching_pool():
    sched = SeedScheduler(run_seed=1, scenario_cfg=make_cfg(), mode="train")
    sched.validate_explicit_seed(50)  # no raise


def test_validate_explicit_seed_rejects_wrong_pool():
    sched = SeedScheduler(run_seed=1, scenario_cfg=make_cfg(), mode="train")
    with pytest.raises(SeedPoolViolation):
        sched.validate_explicit_seed(220)  # test-range seed handed to a train-mode scheduler


def test_validate_explicit_seed_rejects_seed_outside_all_pools():
    sched = SeedScheduler(run_seed=1, scenario_cfg=make_cfg(), mode="train")
    with pytest.raises(SeedPoolViolation):
        sched.validate_explicit_seed(99999)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        SeedScheduler(run_seed=1, scenario_cfg=make_cfg(), mode="bogus")
