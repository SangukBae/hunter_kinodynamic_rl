#!/usr/bin/env python3
"""Standalone profile validator -- no ROS, no Gazebo required.

    python3 -m hunter_kinodynamic_rl.config.validation kinodynamic_tqc

Mirrors drl_experiments' ``run_profile.py --validate-only`` pattern: this is
the first thing a new profile should pass before anyone tries to launch it.
"""

from __future__ import annotations

import argparse
import sys

from hunter_kinodynamic_rl.config.loader import load_profile
from hunter_kinodynamic_rl.config.schema import ConfigError


def validate(profile_name: str, config_root: str = None) -> int:
    try:
        profile = load_profile(profile_name, config_root=config_root)
    except ConfigError as e:
        print(f"INVALID profile {profile_name!r}: {e}", file=sys.stderr)
        return 1
    print(f"OK profile={profile.name}")
    print(f"  robot={profile.robot.name} wheelbase={profile.robot.wheelbase_m}m "
          f"steering_limit={profile.robot.steering_limit_deg}deg "
          f"kappa_max={profile.robot.max_curvature:.4f} 1/m")
    print(f"  action_space.mode={profile.action_space.mode}")
    print(f"  features={profile.features}")
    print(f"  risk.enabled={profile.risk.enabled} counterfactual.enabled={profile.counterfactual.enabled}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", help="profile name (config/profiles/<name>.yaml) or path to a YAML file")
    parser.add_argument("--config-root", default=None, help="override the config/ root directory")
    args = parser.parse_args(argv)
    return validate(args.profile, config_root=args.config_root)


if __name__ == "__main__":
    raise SystemExit(main())
