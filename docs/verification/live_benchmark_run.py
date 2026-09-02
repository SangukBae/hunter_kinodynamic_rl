#!/usr/bin/env python3
"""Live A/B benchmark run (item 11.9) -- small scenario count/option budget
for a wiring/correctness smoke run (not a statistical performance claim),
using the real frozen Local TQC checkpoint over live Gazebo. No trained
Global checkpoint exists yet, so ablation B evaluates the built-in
goal-seeking heuristic (documented, not silently claimed as a Global DQN
comparison) -- this run's purpose is to confirm the benchmark PIPELINE
(clock domain, telemetry, provenance) is correct end-to-end, per the task's
own "wiring/metric correctness와 policy 성능을 분리해 보고" instruction.
"""
import rclpy

from hunter_kinodynamic_rl.evaluation.run_live_hierarchical_benchmark import run_live_hierarchical_benchmark

rclpy.init()
try:
    result = run_live_hierarchical_benchmark(
        "hierarchical_phase4", num_scenarios=3, seed=20000, mode="test",
        global_checkpoint_dir="", global_checkpoint_name="final",
        output_dir="runtime/hierarchical_benchmark", logger=print,
        max_options=3, max_local_steps=10,
    )
    import json
    print("RESULT_SUMMARY_JSON=" + json.dumps(result, default=str))
finally:
    if rclpy.ok():
        rclpy.shutdown()
