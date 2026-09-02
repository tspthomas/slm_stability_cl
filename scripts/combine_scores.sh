#/bin/bash

experiments=(
  "qwen35_08b_paper"
  "qwen35_08b_paper_321"
  "llama32_1b_paper"
  "llama32_1b_paper_321"
  "gemma3_1b_paper"
  "gemma3_1b_paper_321"
)

for experiment_name in "${experiments[@]}"; do
  echo -e "\033[36mProcessing: ${experiment_name}\033[0m"
  experiment_output_dir="results/${experiment_name}/cl_metrics"

  uv run scripts/combine_scores.py \
    --run-dir outputs/${experiment_name} \
    --output-dir ${experiment_output_dir}
done
