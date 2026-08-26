#!/usr/bin/env python3
"""Vanilla TQC trainer -- ``ros2 run hunter_kinodynamic_rl train_node.py``
resolves here when ``features.risk_critic=false`` (baseline_tqc,
legacy_waypoint_tqc, kinodynamic_tqc, kinodynamic_tqc_temporal profiles)."""

from __future__ import annotations

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.rl.algorithms.tqc.agent import Agent
from hunter_kinodynamic_rl.training.trainer_base import TrainerBase


class TQCTrainer(TrainerBase):
    def build_agent(self) -> Agent:
        return Agent(self.state_dim, self.action_dim, self.max_action, self.profile.hyperparameters)

    def agent_train_step(self, batch):
        return self.agent.train_step(batch["state"], batch["action"], batch["next_state"],
                                      batch["reward"], batch["not_done"])


def main(profile_name: str = "kinodynamic_tqc", run_root: str = "runtime/experiments",
         resume: bool = False, resume_run_dir: str = None, resume_checkpoint_tag: str = "latest"):
    profile = load_profile(profile_name)
    if profile.runtime.deployment == "real_hardware":
        raise SystemExit(
            f"profile {profile_name!r} is marked runtime.deployment=real_hardware (inference-only) "
            "-- refusing to train against it. Training happens only in simulation (section P2)."
        )
    trainer = TQCTrainer(profile, run_root, resume=resume, resume_run_dir=resume_run_dir,
                          resume_checkpoint_tag=resume_checkpoint_tag)
    failed = True
    try:
        result = trainer.run()
        failed = False
        return result
    finally:
        trainer.shutdown(failed=failed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="kinodynamic_tqc")
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--resume-checkpoint-tag", default="latest")
    args, _ = parser.parse_known_args()
    main(args.profile, resume=bool(args.resume_run_dir), resume_run_dir=args.resume_run_dir,
         resume_checkpoint_tag=args.resume_checkpoint_tag)
