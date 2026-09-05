# Continual Learning for Sequential Personalization of Small Language Models

This repository contains the experimental code for the paper "Continual
Learning for Sequential Personalization of Small Language Models: A Stability
Monitoring Analysis." The experiments train causal language models across a
fixed order of tasks and evaluate both task accuracy and reference-set stability
after each checkpoint.

The code is intended to make the paper experiments easier to inspect and rerun.
It is deliberately lightweight: configuration files define the model, task
order, training settings, evaluation splits, and output locations.

## Paper

[Continual Learning for Sequential Personalization of Small Language Models: A
Stability Monitoring Analysis](https://arxiv.org/abs/2606.27634)

## Result Versions

- `results/v1/` contains the results associated with the original paper and
  arXiv v1.
- `results/v2/` contains the corrected stability results used by the revised
  manuscript intended for arXiv v2.

The correction affects KL divergence, entropy, and margin. Training, task
accuracy, and continual-learning metrics are unaffected. The implementation
fix and regression tests are documented in
[PR #3](https://github.com/tspthomas/slm_stability_cl/pull/3).

## What Is Included

- Sequential supervised fine-tuning over three tasks: FOMC, ScienceQA, and
  NumGLUE-cm.
- LoRA-based parameter-efficient fine-tuning for supported model configs.
- Evaluation after the base model and after each task checkpoint.
- A mixed-task reference set for measuring stability across checkpoints.
- Analysis scripts for combining task scores and summarizing stability metrics.

## Repository Layout

```text
src/
  config/              Paper experiment configurations
  constants.py         Task prompts and shared defaults
  data.py              Chat SFT dataset and collator utilities
  evaluation.py          Accuracy and reference-set evaluation
  stability.py         Reference stability metrics
  train.py             LoRA setup and per-task training loop
  execute_training.py  Main experiment entry point

data/
  llm-cl-*/            Task splits and reference sets

results/
  v1/                  Original paper and arXiv v1 results
  v2/                  Corrected results for the revised manuscript

scripts/
  combine_scores.py        Aggregate accuracy CSVs
  analyze_stability.py     Aggregate stability CSVs
  build_reference_set.py   Build reference splits from validation data

tests/
  Unit tests for utility, data, training, and evaluation helpers
```

## Setup

The project uses `uv` for dependency management.

```bash
uv sync
```

The configured experiments use Hugging Face models, so a run may require model
access, network access for initial downloads, and a CUDA/MPS-capable machine
depending on the selected model.

## Running an Experiment

Choose one of the paper configs in `src/config/`, then run:

```bash
uv run python src/execute_training.py --config src/config/qwen_paper.yaml
```

Other available configs include Qwen, Llama, and Gemma variants, with alternate
task orders marked by the `_321` suffix.

Experiment outputs are written to the configured `experiment.output_dir`, for
example:

```text
outputs/qwen35_08b_paper/
  scores_seed_*.csv
  eval_*/
  reference_seed_*/
  stability_seed_*/
  model_*_lora/
```

## Analyzing Results

Aggregate task accuracy across seeds:

```bash
uv run scripts/combine_scores.py \
  --run-dir outputs/qwen35_08b_paper \
  --output-dir results/v2/qwen35_08b_paper/cl_metrics
```

Compute continual-learning summary metrics from the combined scores:

```bash
uv run scripts/compute_metrics.py \
  --scores-csv results/v2/qwen35_08b_paper/cl_metrics/scores_all.csv \
  --task-order task_1,task_2,task_3 \
  --output-dir results/v2/qwen35_08b_paper/cl_metrics
```

This writes per-seed and summary CSVs for metrics such as overall performance,
backward transfer, forgetting, forward transfer, adaptation gain, and final
checkpoint performance.

Aggregate stability metrics:

```bash
uv run scripts/analyze_stability.py \
  --input outputs/qwen35_08b_paper \
  --output-dir results/v2/qwen35_08b_paper/stability_metrics
```

The `scripts/` directory also includes small `.sh` wrappers showing the command
sequence used for common analysis runs. Treat them as editable templates: check
the `experiment_name` and task order before launching them.

## Tests

Install the project into the local `uv` environment before running tests:

```bash
uv sync
```

The current test suite uses Python's built-in `unittest` runner:

```bash
uv run python -m unittest discover
```

These tests focus on deterministic helper behavior, data formatting, training
loop mechanics, evaluation bookkeeping, and file-output formats. They avoid
loading real models or running full training jobs.

## Citation

```bibtex
@article{paula2026continual,
  title   = {Continual Learning for Sequential Personalization of Small Language Models: A Stability Monitoring Analysis},
  author  = {Paula, Thomas S. and Kupssinsk{\"u}, Lucas S. and Barros, Rodrigo C.},
  journal = {arXiv preprint arXiv:2606.27634},
  year    = {2026}
}
```
