#!/usr/bin/env python3
"""Risk-aware kinodynamic TQC trainer -- resolved when
``features.risk_critic=true`` (kinodynamic_tqc_risk_supervised_only,
kinodynamic_tqc_risk, kinodynamic_tqc_counterfactual profiles). Privileged risk labels arrive via
environment_node.py's risk-telemetry side channel
(env/simulation/risk_telemetry.py), transported end-to-end by
TrainerBase/EnvironmentClient -- see docs/ARCHITECTURE.md's "Training vs
inference" note for why this label is training-only."""

from __future__ import annotations

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.rl.algorithms.kinodynamic_tqc.agent import Agent
from hunter_kinodynamic_rl.training.trainer_base import TrainerBase


class KinodynamicTQCTrainer(TrainerBase):
    risk_aware = True

    def build_agent(self) -> Agent:
        return Agent(self.state_dim, self.action_dim, self.max_action,
                      self.profile.hyperparameters, self.profile.risk, self.profile.counterfactual)

    def agent_train_step(self, batch):
        return self.agent.train_step(batch)


def main(profile_name: str = "kinodynamic_tqc_risk", run_root: str = "runtime/experiments",
         resume: bool = False, resume_run_dir: str = None, resume_checkpoint_tag: str = "latest"):
    profile = load_profile(profile_name)
    if profile.runtime.deployment == "real_hardware":
        raise SystemExit(
            f"profile {profile_name!r} is marked runtime.deployment=real_hardware (inference-only) "
            "-- refusing to train against it. Training happens only in simulation (section P2)."
        )
    trainer = KinodynamicTQCTrainer(profile, run_root, resume=resume, resume_run_dir=resume_run_dir,
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
    parser.add_argument("--profile", default="kinodynamic_tqc_risk")
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--resume-checkpoint-tag", default="latest")
    args, _ = parser.parse_known_args()
    main(args.profile, resume=bool(args.resume_run_dir), resume_run_dir=args.resume_run_dir,
         resume_checkpoint_tag=args.resume_checkpoint_tag)
