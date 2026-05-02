#/bin/bash

experiment_name="qwen35_08b_cl"
experiment_output_dir="results/${experiment_name}/stability_metrics"

uv run scripts/analyze_stability.py \
  --input outputs/${experiment_name} \
  --output-dir ${experiment_output_dir}