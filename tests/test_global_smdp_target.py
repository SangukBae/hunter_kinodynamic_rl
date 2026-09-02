"""Phase 4 Global RL: SMDP target arithmetic (plan section 8.8) --

    target = option_reward + gamma^local_steps * (1 - mission_done) * max_valid Q_target(next_state)

Torch-gated -- skips cleanly when torch is not installed."""

import pytest

torch = pytest.importorskip("torch")

from hunter_kinodynamic_rl.navigation.global_rl.agent import smdp_target  # noqa: E402


def test_gamma_is_raised_to_local_steps_exactly():
    reward = torch.zeros(3, 1)
    mission_done = torch.zeros(3, 1)
    next_q = torch.ones(3, 1)
    local_steps = torch.tensor([[1.0], [2.0], [5.0]])
    target = smdp_target(reward, 0.9, local_steps, mission_done, next_q)
    expected = torch.tensor([[0.9 ** 1], [0.9 ** 2], [0.9 ** 5]])
    torch.testing.assert_close(target, expected)


def test_longer_options_discount_the_bootstrap_more():
    reward = torch.zeros(1, 1)
    mission_done = torch.zeros(1, 1)
    next_q = torch.ones(1, 1)
    short = smdp_target(reward, 0.99, torch.tensor([[1.0]]), mission_done, next_q)
    long = smdp_target(reward, 0.99, torch.tensor([[50.0]]), mission_done, next_q)
    assert long.item() < short.item()


def test_mission_done_zeroes_the_bootstrap_term():
    reward = torch.tensor([[3.0]])
    local_steps = torch.tensor([[10.0]])
    next_q = torch.tensor([[100.0]])
    done_target = smdp_target(reward, 0.9, local_steps, torch.tensor([[1.0]]), next_q)
    not_done_target = smdp_target(reward, 0.9, local_steps, torch.tensor([[0.0]]), next_q)
    assert torch.isclose(done_target, reward).item()
    assert not_done_target.item() > done_target.item()


def test_reward_alone_survives_regardless_of_bootstrap():
    reward = torch.tensor([[7.5]])
    target = smdp_target(reward, 0.99, torch.tensor([[3.0]]), torch.tensor([[1.0]]), torch.tensor([[999.0]]))
    torch.testing.assert_close(target, reward)


def test_subgoal_success_does_not_gate_bootstrapping_only_mission_done_does():
    """The SMDP target formula bootstraps off ``mission_done`` alone --
    ``subgoal_success``/``subgoal_done`` is a SEPARATE diagnostic field
    (stored in the replay buffer for logging/reward-shaping, never read by
    this function) and must have NO effect on the target. This is the
    "mission_done과 subgoal_done의 bootstrapping 차이" the plan requires be
    tested: a subgoal REACHED mid-mission still bootstraps normally."""
    reward = torch.tensor([[1.0]])
    local_steps = torch.tensor([[4.0]])
    next_q = torch.tensor([[50.0]])
    # subgoal reached (would be subgoal_success=True upstream) but mission NOT done.
    target_subgoal_reached_mission_ongoing = smdp_target(reward, 0.95, local_steps, torch.tensor([[0.0]]), next_q)
    # subgoal FAILED (subgoal_success=False upstream) but mission NOT done either.
    target_subgoal_failed_mission_ongoing = smdp_target(reward, 0.95, local_steps, torch.tensor([[0.0]]), next_q)
    assert torch.isclose(target_subgoal_reached_mission_ongoing, target_subgoal_failed_mission_ongoing)
    # Only a mission_done=1 changes the bootstrap, regardless of what the
    # just-finished subgoal's own outcome was.
    target_mission_done = smdp_target(reward, 0.95, local_steps, torch.tensor([[1.0]]), next_q)
    assert not torch.isclose(target_mission_done, target_subgoal_reached_mission_ongoing)
    assert torch.isclose(target_mission_done, reward)


def test_batched_targets_broadcast_correctly():
    reward = torch.tensor([[1.0], [2.0]])
    local_steps = torch.tensor([[1.0], [2.0]])
    mission_done = torch.tensor([[0.0], [1.0]])
    next_q = torch.tensor([[10.0], [10.0]])
    target = smdp_target(reward, 0.5, local_steps, mission_done, next_q)
    torch.testing.assert_close(target, torch.tensor([[1.0 + 0.5 * 10.0], [2.0]]))
