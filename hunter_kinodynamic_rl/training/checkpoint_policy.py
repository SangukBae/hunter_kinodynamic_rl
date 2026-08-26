"""Periodic-checkpoint cadence gate (section P0-4), extracted into its own
ROS-free module so it is directly host-testable (``trainer_base.py`` itself
imports ``rclpy`` at module scope and cannot be imported on a bare
host checkout -- see CLAUDE.md's Testing section).
"""

from __future__ import annotations


def checkpoint_due(step: int, last_checkpoint_step: int, eval_freq: int) -> bool:
    """True once at least ``eval_freq`` steps have elapsed since the last
    periodic checkpoint. ONLY ever called from ``TrainerBase.run()``'s
    ``if done:`` block -- right after an episode finishes, BEFORE the next
    episode's ``_new_episode()`` (which draws the next seed and calls
    ``env.reset()``) -- never mid-episode.

    Why this matters for deterministic resume: environment_node.py's own
    per-episode state (frame stack, prev_action, prev_goal_distance,
    obstacle positions, command-delay queue, domain-randomization draw) is
    NEVER checkpointed -- it is entirely rebuilt from scratch by the next
    ``env.reset()`` call, which is DETERMINISTIC given only the episode's
    seed. ``run()``'s architecture already calls ``_new_episode()`` exactly
    once, unconditionally, before its main loop starts -- both on a fresh
    run (the first episode) and on a resumed run (the next episode after
    the restored checkpoint). So as long as a checkpoint is ALWAYS taken at
    a point where "the next ``_new_episode()`` call has not happened yet"
    -- i.e. an episode boundary -- resuming reproduces the identical
    ``env.reset()`` outcome (same seed) the uninterrupted run would have
    produced next, with no Gazebo/observation/scenario state to restore at
    all.

    A checkpoint taken MID-episode instead (the previous
    ``step % eval_freq == 0`` gate, which could land anywhere) would save a
    ``seed_scheduler`` state that has ALREADY consumed the current
    episode's seed, but a ``global_step`` that is still INSIDE that
    episode -- resume's unconditional ``_new_episode()`` call would then
    draw the NEXT seed and abandon the rest of the interrupted episode
    outright, silently diverging from what the uninterrupted run actually
    did for the remainder of that episode."""
    return step - last_checkpoint_step >= eval_freq
