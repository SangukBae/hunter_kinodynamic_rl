#!/usr/bin/env python3
"""Live Global-training + resume smoke test (item 11.8) -- live=True drives
the REAL LiveGazeboLocalExecutor. Small mission counts (wall-clock cost is
real): 2 missions, save, close; then resume for 2 more missions live,
confirming the resume fail-fast/RNG-restore machinery (item 4) actually
runs against a live-produced checkpoint, not just a pytest-synthesized one.
"""
import json
import os
import sys

import rclpy

from hunter_kinodynamic_rl.nodes.hierarchical_train_node import train_hierarchical_dqn

rclpy.init()
try:
    run_root = "runtime/hierarchical_experiments_live_smoke"
    print("[live_train] phase 1: 2 missions, live=True", flush=True)
    result1 = train_hierarchical_dqn(
        "hierarchical_phase4", run_root=run_root, num_missions=2, live=True, logger=print,
    )
    print(f"[live_train] phase 1 result: {result1}", flush=True)

    meta_path = os.path.join(result1["run_dir"], "hierarchical_metadata.json")
    with open(meta_path) as f:
        meta1 = json.load(f)
    print(f"[live_train] checkpoint fields: rng_state_path={meta1.get('rng_state_path')} "
          f"rng_state_sha256={meta1.get('rng_state_sha256')} "
          f"hierarchical_architecture_fingerprint={meta1.get('hierarchical_architecture_fingerprint')} "
          f"local_checkpoint_manifest_summary={meta1.get('local_checkpoint_manifest_summary')} "
          f"training_update_cadence={meta1.get('training_update_cadence')} "
          f"total_optimizer_updates={meta1.get('total_optimizer_updates')}", flush=True)

    print("[live_train] phase 2: resume for 2 more missions, live=True", flush=True)
    result2 = train_hierarchical_dqn(
        "hierarchical_phase4", run_root=run_root, num_missions=2,
        resume_run_dir=result1["run_dir"], resume_checkpoint_tag="latest", live=True, logger=print,
    )
    print(f"[live_train] phase 2 (resumed) result: {result2}", flush=True)
    assert result2["global_step"] > result1["global_step"], "resume did not advance global_step"
    print("[live_train] PASS", flush=True)
finally:
    if rclpy.ok():
        rclpy.shutdown()
