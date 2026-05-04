#/bin/bash

#experiment_name="qwen35_08b_cl_paper"
#experiment_name="qwen35_08b_paper_321"
#experiment_name="llama32_1b_cl_paper"
#experiment_name="llama32_1b_cl_paper_321"
#experiment_name="gemma3_1b_cl_paper"
experiment_name="gemma3_1b_cl_paper_321"
experiment_output_dir="results/${experiment_name}/cl_metrics"

uv run scripts/combine_scores.py \
  --run-dir outputs/${experiment_name} \
  --output-dir ${experiment_output_dir}
