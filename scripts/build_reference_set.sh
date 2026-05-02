#/bin/bash

uv run scripts/build_reference_set.py \
  --dataset-root data/llm-cl-5000 \
  --val-filename eval.json \
  --ref-fraction 0.2 \
  --seed 33 \
  --overwrite