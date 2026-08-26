#!/usr/bin/env python3
"""Ablation-A baseline: legacy [r, theta, yield] waypoint action, vanilla
TQC. Not a separate implementation -- ``action_space.mode: legacy_waypoint``
(config/profiles/legacy_waypoint_tqc.yaml) already makes
trajectory/action_space.py + trajectory/trajectory_executor.py emit the same
waypoint contract drl_agent's hybrid action uses, so this baseline runs
through the exact SAME trainer/agent code as every other profile (section
36: ablations are config flags, never a forked code path). This file exists
only as the section-34-mandated, directly-named entrypoint.
"""

from hunter_kinodynamic_rl.training.train_tqc import main as _train_main


def main(run_root: str = "runtime/experiments"):
    return _train_main(profile_name="legacy_waypoint_tqc", run_root=run_root)


if __name__ == "__main__":
    main()
