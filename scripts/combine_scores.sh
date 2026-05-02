#/bin/bash

experiment_name="qwen35_08b_cl"
experiment_output_dir="results/${experiment_name}/cl_metrics"

uv run scripts/combine_scores.py \
  --run-dir outputs/${experiment_name} \
  --output-dir ${experiment_output_dir}
