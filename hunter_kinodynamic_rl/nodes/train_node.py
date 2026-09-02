#!/usr/bin/env python3
"""``ros2 run hunter_kinodynamic_rl train_node.py --ros-args -p profile:=kinodynamic_tqc_risk``

Resolves the profile, then dispatches to the vanilla TQC, risk-aware TQC, or
vanilla SAC trainer based on ``algorithm.name``/``features.risk_critic`` --
mirrors drl_agent's train_node.py resolve-then-exec pattern (CLAUDE.md's
Package Structure section) at the scale this package needs (three trainer
variants, not a registry).
"""

import argparse
import sys

from hunter_kinodynamic_rl.config.loader import load_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", "-p", dest="profile_kv", action="append", default=[])
    args, _ = parser.parse_known_args()

    profile_name = "kinodynamic_tqc"
    resume_run_dir = None
    resume_checkpoint_tag = "latest"
    for kv in args.profile_kv:
        if kv.startswith("profile:="):
            profile_name = kv.split(":=", 1)[1]
        elif kv.startswith("resume_run_dir:="):
            resume_run_dir = kv.split(":=", 1)[1]
        elif kv.startswith("resume_checkpoint_tag:="):
            resume_checkpoint_tag = kv.split(":=", 1)[1]

    profile = load_profile(profile_name)
    if profile.algorithm.name == "sac":
        from hunter_kinodynamic_rl.training.train_sac import main as train_main
    elif profile.features.risk_critic:
        from hunter_kinodynamic_rl.training.train_kinodynamic_tqc import main as train_main
    else:
        from hunter_kinodynamic_rl.training.train_tqc import main as train_main
    train_main(profile_name, resume=bool(resume_run_dir), resume_run_dir=resume_run_dir,
               resume_checkpoint_tag=resume_checkpoint_tag)


if __name__ == "__main__":
    main()
