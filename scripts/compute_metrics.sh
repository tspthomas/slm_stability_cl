#/bin/bash

experiments=(
  "qwen35_08b_paper"
  "qwen35_08b_paper_321"
  "llama32_1b_paper"
  "llama32_1b_paper_321"
  "gemma3_1b_paper"
  "gemma3_1b_paper_321"
)

echo -e "\033[36mComputing metrics for experiments (order 1,2,3)\033[0m"
for experiment_name in "${experiments[@]}"; do
  echo -e "\033[36mProcessing: ${experiment_name}\033[0m"
  experiment_output_dir="results/${experiment_name}/cl_metrics"

  uv run scripts/compute_metrics.py \
    --scores-csv ${experiment_output_dir}/scores_all.csv \
    --task-order task_1,task_2,task_3
done


experiments_321=(
  # "qwen35_08b_paper_321_kl_fixed"
  # "llama32_1b_cl_paper_321_kl_fixed"
  # "gemma3_1b_cl_paper_321_kl_fixed"
  "llama32_1b_paper_321_date_fixed"
)

echo -e "\033[36mComputing metrics for experiments (order 3,2,1)\033[0m"
for experiment_name in "${experiments_321[@]}"; do
  echo -e "\033[36mProcessing: ${experiment_name}\033[0m"
  experiment_output_dir="results/${experiment_name}/cl_metrics"

  uv run scripts/compute_metrics.py \
    --scores-csv ${experiment_output_dir}/scores_all.csv \
    --task-order task_3,task_2,task_1
done
