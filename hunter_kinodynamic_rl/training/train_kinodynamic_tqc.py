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
from hunter_kinodynamic_rl.training.preflight import _format_report, run_local_preflight
from hunter_kinodynamic_rl.training.trainer_base import TrainerBase


class KinodynamicTQCTrainer(TrainerBase):
    risk_aware = True

    def build_agent(self) -> Agent:
        return Agent(self.state_dim, self.action_dim, self.max_action,
                      self.profile.hyperparameters, self.profile.risk, self.profile.counterfactual)

    def agent_train_step(self, batch):
        return self.agent.train_step(batch)


def main(profile_name: str = "kinodynamic_tqc_risk", run_root: str = "runtime/experiments",
         resume: bool = False, resume_run_dir: str = None, resume_checkpoint_tag: str = "latest",
         dry_run: bool = False, skip_preflight: bool = False):
    """``dry_run=True`` (requirement A's preflight gate, ``--dry-run``/
    ``--validate-only`` on the CLI): runs ``training.preflight.run_local_preflight``
    and returns its report WITHOUT touching Gazebo/ROS or starting training
    -- never a partial/implicit training start.

    Defect-fix item 11: a REAL training start (``dry_run=False``) now runs
    the SAME preflight gate automatically first (never skipped implicitly)
    -- the historical gap was that preflight only ever ran when a caller
    remembered to pass ``--dry-run`` separately; a direct launch never
    validated anything before starting a (potentially multi-hour) training
    run. ``skip_preflight=True`` (``--skip-preflight`` on the CLI) is the
    explicit, logged bypass for a caller that has already validated the
    profile out-of-band and wants to skip the (cheap, but nonzero-cost --
    it empirically samples the scenario generator) recheck."""
    if dry_run:
        report = run_local_preflight(profile_name, run_root=run_root, resume=resume, resume_run_dir=resume_run_dir)
        print(_format_report(report))
        return report
    if skip_preflight:
        print(
            f"[train_kinodynamic_tqc] WARNING: skip_preflight=True -- proceeding to train profile "
            f"{profile_name!r} WITHOUT running requirement-A preflight checks first"
        )
    else:
        preflight_report = run_local_preflight(
            profile_name, run_root=run_root, resume=resume, resume_run_dir=resume_run_dir,
        )
        if not preflight_report.ok:
            print(_format_report(preflight_report))
            preflight_report.raise_if_failed()
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
    parser.add_argument("--dry-run", "--validate-only", dest="dry_run", action="store_true",
                         help="run requirement-A preflight checks and exit without training")
    parser.add_argument("--skip-preflight", action="store_true",
                         help="skip the automatic requirement-A preflight gate before a real training start "
                              "(logged explicitly -- never a silent skip)")
    args, _ = parser.parse_known_args()
    result = main(args.profile, resume=bool(args.resume_run_dir), resume_run_dir=args.resume_run_dir,
                  resume_checkpoint_tag=args.resume_checkpoint_tag, dry_run=args.dry_run,
                  skip_preflight=args.skip_preflight)
    if args.dry_run:
        raise SystemExit(0 if result.ok else 1)
