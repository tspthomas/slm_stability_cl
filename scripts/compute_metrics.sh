#/bin/bash

experiment_name="qwen35_08b_cl"
experiment_results_dir="results/${experiment_name}/cl_metrics"

uv run scripts/compute_metrics.py \
  --scores-csv ${experiment_results_dir}/scores_all.csv \
  --task-order task_1,task_2,task_3
