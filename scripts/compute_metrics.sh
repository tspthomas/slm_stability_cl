#/bin/bash

#experiment_name="qwen35_08b_cl_paper"
#experiment_name="qwen35_08b_paper_321"
#experiment_name="llama32_1b_cl_paper"
#experiment_name="llama32_1b_cl_paper_321"
#experiment_name="gemma3_1b_cl_paper"
experiment_name="gemma3_1b_cl_paper_321"
experiment_results_dir="results/${experiment_name}/cl_metrics"

uv run scripts/compute_metrics.py \
  --scores-csv ${experiment_results_dir}/scores_all.csv \
  --task-order task_3,task_2,task_1
  #--task-order task_1,task_2,task_3
