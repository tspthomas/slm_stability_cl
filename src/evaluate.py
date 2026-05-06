"""Evaluation utilities for task accuracy and reference-set scoring.

This module renders chat prompts, runs batched generation, normalizes model
outputs, and writes per-task/reference evaluation artifacts. The lower-level
helpers are intentionally small so generation behavior such as EOS handling and
sampling configuration can be tested independently.
"""

import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from constants import (
    GENERATION_DO_SAMPLE,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    GENERATION_USE_CACHE,
)
from data import build_messages, get_task_prompt
from utils import normalize_answer


def batch_iter(dataset: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield fixed-size batches from an indexable dataset.

    Args:
        dataset: Indexable dataset or sequence.
        batch_size: Maximum number of examples per yielded batch.

    Yields:
        Lists of dataset rows. The final batch may be smaller than
        ``batch_size``.
    """
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield [dataset[i] for i in range(start, end)]


def _eos_found(
    generated_ids: torch.Tensor,
    eos_token_id: int | list[int] | None,
) -> bool:
    """Return whether any EOS token appears in generated token ids.

    Args:
        generated_ids: One generated token-id sequence.
        eos_token_id: Single EOS id, multiple EOS ids, or ``None``.

    Returns:
        ``True`` if any configured EOS id occurs in ``generated_ids``.
    """
    if eos_token_id is None:
        return False

    if isinstance(eos_token_id, list):
        return any((generated_ids == eos_id).any().item() for eos_id in eos_token_id)

    return (generated_ids == eos_token_id).any().item()


def build_generation_text(
    tokenizer: Any,
    example: dict[str, Any],
    task_prompt: str,
    system_prompt: str,
    use_system_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    """Render one dataset example into a generation prompt string.

    Args:
        tokenizer: Tokenizer implementing ``apply_chat_template``.
        example: Dataset row containing a ``prompt`` field.
        task_prompt: Task-level instruction prepended to the raw prompt.
        system_prompt: Optional system message text.
        use_system_prompt: Whether to include the system message.
        enable_thinking: Passed through to chat-template rendering.

    Returns:
        Rendered chat-template string ending with an assistant generation
        prompt.
    """
    messages = build_messages(
        prompt=example["prompt"],
        task_prompt=task_prompt,
        system_prompt=system_prompt,
        use_system_prompt=use_system_prompt,
    )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def get_pad_token_id(tokenizer: Any) -> int:
    """Return a usable padding token id for generation.

    Args:
        tokenizer: Tokenizer with ``pad_token_id`` and/or ``eos_token_id``.

    Returns:
        The tokenizer pad token id, falling back to EOS when padding is absent.

    Raises:
        ValueError: If neither pad nor EOS token id is available.
    """
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")


def get_eos_token_ids(tokenizer: Any, model: Any | None = None) -> list[int] | None:
    """Collect all EOS token ids relevant for chat generation.

    Args:
        tokenizer: Tokenizer with EOS metadata and token-id conversion.
        model: Optional model whose generation config may define extra EOS ids.

    Returns:
        Ordered unique EOS ids from the tokenizer, model generation config, and
        common chat end markers. Returns ``None`` when no ids are available.
    """
    eos_ids: list[int] = []

    def add_id(x: int | list[int] | None) -> None:
        if x is None:
            return
        if isinstance(x, list):
            for v in x:
                add_id(v)
        elif x not in eos_ids:
            eos_ids.append(x)

    add_id(getattr(tokenizer, "eos_token_id", None))

    if model is not None:
        add_id(getattr(model.generation_config, "eos_token_id", None))

    for tok in ["<end_of_turn>", "<|im_end|>"]:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if (
            tok_id is not None
            and tok_id != tokenizer.unk_token_id
            and tok_id not in eos_ids
        ):
            eos_ids.append(tok_id)

    return eos_ids if eos_ids else None


def trim_after_eos(
    generated_ids: torch.Tensor,
    eos_token_id: int | list[int] | None,
) -> torch.Tensor:
    """Trim generated ids after the first EOS token, keeping that EOS token.

    Args:
        generated_ids: One generated token-id sequence.
        eos_token_id: Single EOS id, multiple EOS ids, or ``None``.

    Returns:
        ``generated_ids`` up to and including the first EOS token. If no EOS id
        is configured or found, returns the original tensor.
    """
    if eos_token_id is None:
        return generated_ids

    eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]

    for idx, token_id in enumerate(generated_ids.tolist()):
        if token_id in eos_ids:
            return generated_ids[: idx + 1]

    return generated_ids


def build_generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build keyword arguments for ``model.generate`` from config.

    Args:
        config: Experiment config that may contain a ``generation`` section.

    Returns:
        Generation keyword arguments. Sampling-only parameters such as
        temperature and top-p are included only when sampling is enabled.
    """
    generation_cfg = config.get("generation", {})
    do_sample = generation_cfg.get("do_sample", GENERATION_DO_SAMPLE)

    kwargs = {
        "do_sample": do_sample,
        "use_cache": generation_cfg.get("use_cache", GENERATION_USE_CACHE),
        "num_beams": generation_cfg.get("num_beams", 1),
    }

    if do_sample:
        kwargs["temperature"] = generation_cfg.get(
            "temperature", GENERATION_TEMPERATURE
        )
        kwargs["top_p"] = generation_cfg.get("top_p", GENERATION_TOP_P)

    return kwargs


def generate_batch_from_texts(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    texts: list[str],
    device: torch.device | str,
    max_new_tokens: int,
) -> list[tuple[str, bool]]:
    """Generate completions for already-rendered chat-template strings.

    The tokenizer is temporarily switched to left padding so decoder-only
    models generate from the rightmost prompt token. The original padding side
    is restored before returning, even if tokenization fails.

    Args:
        model: Causal language model with a ``generate`` method.
        tokenizer: Tokenizer used for padding, tokenization, and decoding.
        config: Experiment config containing optional generation settings.
        texts: Rendered prompt strings.
        device: Device where tokenized inputs should be moved.
        max_new_tokens: Maximum number of tokens to generate per prompt.

    Returns:
        ``(decoded_text, hit_max_tokens)`` pairs. ``hit_max_tokens`` is true
        when generation reached ``max_new_tokens`` without emitting EOS.
    """
    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_ids = get_eos_token_ids(tokenizer, model)

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
    finally:
        tokenizer.padding_side = old_padding_side

    prompt_length = inputs["input_ids"].shape[1]
    generation_kwargs = build_generation_kwargs(config)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_ids,
        )

    generated_batch = outputs[:, prompt_length:]

    results = []
    for generated_ids in generated_batch:
        ended_with_eos = _eos_found(generated_ids, eos_token_ids)
        hit_max_tokens = len(generated_ids) >= max_new_tokens and not ended_with_eos

        trimmed_ids = trim_after_eos(generated_ids, eos_token_ids)

        text = tokenizer.decode(
            trimmed_ids,
            skip_special_tokens=False,
        ).strip()

        results.append((text, hit_max_tokens))

    return results


def save_eval_results(
    val_metrics: dict[str, Any],
    output_dir: str,
    task: str,
    seed: int,
    split_name: str = "val",
) -> str:
    """Write per-example predictions and aggregate metrics for one eval split.

    Args:
        val_metrics: Metrics dictionary from ``evaluate_accuracy``. The
            optional ``examples`` key is written as JSONL and excluded from the
            aggregate metrics JSON.
        output_dir: Root experiment output directory.
        task: Task label embedded in the output directory name.
        seed: Experiment seed embedded in the output directory name.
        split_name: Split prefix used for prediction and metrics filenames.

    Returns:
        Directory containing the written evaluation artifacts.
    """
    eval_output_dir = os.path.join(output_dir, f"eval_{task}_{seed}")
    os.makedirs(eval_output_dir, exist_ok=True)

    predictions_path = os.path.join(eval_output_dir, f"{split_name}_predictions.jsonl")
    metrics_path = os.path.join(eval_output_dir, f"{split_name}_metrics.json")

    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in val_metrics.get("examples", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_without_examples = {k: v for k, v in val_metrics.items() if k != "examples"}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_without_examples, f, indent=2, ensure_ascii=False)

    return eval_output_dir


def append_score_rows(
    output_dir: str,
    rows: list[dict[str, Any]],
    filename: str = "scores.csv",
) -> None:
    """Append task-level score rows to a cumulative CSV file.

    Args:
        output_dir: Directory where the CSV file should live.
        rows: Flat score dictionaries matching the configured CSV fieldnames.
        filename: CSV filename relative to ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "step",
                "checkpoint_task",
                "eval_task",
                "eval_task_name",
                "accuracy",
                "correct",
                "total",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def evaluate_checkpoint_on_all_tasks(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    task_order: list[str],
    system_prompt: str,
    use_system_prompt: bool,
    step: int,
    checkpoint_task: str,
    seed: int,
    device: torch.device | str,
) -> list[dict[str, Any]] | None:
    """Evaluate one checkpoint on every configured task validation split.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer used for prompt rendering and generation.
        config: Experiment configuration.
        task_order: Ordered task ids from the continual-learning schedule.
        system_prompt: Optional system prompt content.
        use_system_prompt: Whether system prompts are included in rendered
            messages.
        step: Continual-learning step for this checkpoint.
        checkpoint_task: Task id or ``"base"`` identifying the checkpoint.
        seed: Experiment seed.
        device: Device used for generation.

    Returns:
        Rows appended to the score CSV, or ``None`` when main eval-set
        evaluation is disabled.
    """
    if not config.get("eval_set_evaluation", {}).get("enabled", False):
        print(
            "Skipping evaluation on main validation sets as eval_set_evaluation is disabled in config."
        )
        return None

    rows = []

    model.eval()
    model.config.use_cache = True

    split_file_key = config["eval_set_evaluation"].get("split_file_key", "val_file")
    for eval_task in task_order:
        task_config = config["tasks"][eval_task]
        task_name = task_config["name"]

        print(
            f"Evaluating step={step}, checkpoint={checkpoint_task}, "
            f"eval_task={eval_task}-{task_name},"
            f"split_file_key={split_file_key}"
        )

        task_prompt = get_task_prompt(task_name)

        metrics = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            config=config,
            data_file=task_config[split_file_key],
            task_name=task_name,
            task_prompt=task_prompt,
            system_prompt=system_prompt,
            use_system_prompt=use_system_prompt,
            device=device,
            max_examples=config["eval_set_evaluation"].get("max_examples"),
            max_new_tokens=config["eval_set_evaluation"].get("max_new_tokens"),
        )

        rows.append(
            {
                "seed": seed,
                "step": step,
                "checkpoint_task": checkpoint_task,
                "eval_task": eval_task,
                "eval_task_name": task_name,
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
            }
        )

        save_eval_results(
            val_metrics=metrics,
            output_dir=config["experiment"]["output_dir"],
            task=f"step_{step}_{checkpoint_task}_eval_{eval_task}",
            seed=seed,
            split_name=split_file_key.replace("_file", ""),
        )

    append_score_rows(
        output_dir=config["experiment"]["output_dir"],
        rows=rows,
        filename=f"scores_seed_{seed}.csv",
    )

    return rows


def evaluate_accuracy(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    data_file: str,
    task_name: str,
    task_prompt: str,
    system_prompt: str,
    use_system_prompt: bool,
    device: torch.device | str,
    max_examples: int | None = None,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Compute exact-match accuracy for one dataset file.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer used for prompt rendering and generation.
        config: Experiment configuration.
        data_file: JSON dataset file containing ``prompt`` and ``answer``.
        task_name: Normalization task name.
        task_prompt: Task-level instruction prompt.
        system_prompt: Optional system prompt content.
        use_system_prompt: Whether system prompts are included in rendered
            messages.
        device: Device used for generation.
        max_examples: Optional cap for smoke-test evaluation.
        max_new_tokens: Maximum tokens generated per example.
        batch_size: Generation batch size.

    Returns:
        Aggregate accuracy metrics plus an ``examples`` list with per-example
        predictions.
    """
    dataset = load_dataset("json", data_files=data_file, split="train")

    if max_examples is not None:
        print(f"Limiting evaluation to max_examples={max_examples}")
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    model.eval()
    model.config.use_cache = True

    correct = 0
    total = 0
    rows: list[dict[str, Any]] = []

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for examples in tqdm(
        batch_iter(dataset, batch_size),
        total=num_batches,
        desc=f"Evaluating {task_name}",
    ):
        texts = [
            build_generation_text(
                tokenizer=tokenizer,
                example=example,
                task_prompt=task_prompt,
                system_prompt=system_prompt,
                use_system_prompt=use_system_prompt,
                enable_thinking=False,
            )
            for example in examples
        ]

        predictions = generate_batch_from_texts(
            model=model,
            tokenizer=tokenizer,
            config=config,
            texts=texts,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        for example, (prediction_text, hit_max_tokens) in zip(examples, predictions):
            gold = normalize_answer(example["answer"], task_name)
            pred = normalize_answer(prediction_text, task_name)

            is_correct = pred == gold
            correct += int(is_correct)
            total += 1

            rows.append(
                {
                    "prompt": example["prompt"],
                    "gold": gold,
                    "prediction": pred,
                    "raw_prediction": prediction_text,
                    "hit_max_tokens": hit_max_tokens,
                    "correct": is_correct,
                }
            )

    accuracy = correct / total if total > 0 else 0.0

    return {
        "task_name": task_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "examples": rows,
    }


def resolve_reference_task_name(
    example: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Resolve a reference-set row to a normalized task name.

    Args:
        example: Reference-set row. It may include either ``task_name`` or a
            ``task_id`` that appears in ``config["tasks"]``.
        config: Experiment configuration containing task metadata.

    Returns:
        Lowercase task name used for prompting and answer normalization.

    Raises:
        ValueError: If neither a task name nor a resolvable task id is present.
    """
    if "task_name" in example and example["task_name"]:
        return str(example["task_name"]).strip().lower()

    if "task_id" in example and example["task_id"] in config["tasks"]:
        return config["tasks"][example["task_id"]]["name"].strip().lower()

    raise ValueError(
        "Reference example must contain either a valid 'task_name' "
        "or a 'task_id' found in config['tasks']."
    )


def _evaluate_reference_set(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    system_prompt: str,
    use_system_prompt: bool,
    step: int,
    checkpoint_task: str,
    seed: int,
    device: torch.device | str,
) -> dict[str, Any]:
    """Evaluate one checkpoint on the mixed-task reference set.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer used for prompt rendering and generation.
        config: Experiment configuration with a ``reference_set_evaluation``
            section.
        system_prompt: Optional system prompt content.
        use_system_prompt: Whether system prompts are included in rendered
            messages.
        step: Continual-learning step for this checkpoint.
        checkpoint_task: Task id or ``"base"`` identifying the checkpoint.
        seed: Experiment seed.
        device: Device used for generation.

    Returns:
        Aggregate reference-set metrics, per-task metrics, and per-example
        predictions.
    """
    reference_file = config["reference_set_evaluation"]["file"]
    dataset = load_dataset("json", data_files=reference_file, split="train")

    max_examples = config["reference_set_evaluation"].get("max_examples")
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    max_new_tokens = config["reference_set_evaluation"].get("max_new_tokens", 16)
    batch_size = config["reference_set_evaluation"].get("batch_size", 4)

    model.eval()
    model.config.use_cache = True

    total = 0
    correct = 0
    rows = []

    per_task = defaultdict(lambda: {"correct": 0, "total": 0})

    num_batches = (len(dataset) + batch_size - 1) // batch_size

    for examples in tqdm(
        batch_iter(dataset, batch_size),
        total=num_batches,
        desc=f"Reference eval step={step}",
    ):
        task_names = [
            resolve_reference_task_name(example, config) for example in examples
        ]

        texts = [
            build_generation_text(
                tokenizer=tokenizer,
                example=example,
                task_prompt=get_task_prompt(task_name),
                system_prompt=system_prompt,
                use_system_prompt=use_system_prompt,
                enable_thinking=False,
            )
            for example, task_name in zip(examples, task_names)
        ]

        predictions = generate_batch_from_texts(
            model=model,
            tokenizer=tokenizer,
            config=config,
            texts=texts,
            device=device,
            max_new_tokens=max_new_tokens,
        )

        for example, task_name, (prediction_text, hit_max_tokens) in zip(
            examples,
            task_names,
            predictions,
        ):
            gold = normalize_answer(example["answer"], task_name)
            pred = normalize_answer(prediction_text, task_name)

            is_correct = pred == gold

            correct += int(is_correct)
            total += 1

            per_task[task_name]["correct"] += int(is_correct)
            per_task[task_name]["total"] += 1

            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "checkpoint_task": checkpoint_task,
                    "task_name": task_name,
                    "prompt": example["prompt"],
                    "gold": gold,
                    "prediction": pred,
                    "raw_prediction": prediction_text,
                    "hit_max_tokens": hit_max_tokens,
                    "correct": is_correct,
                }
            )

    accuracy = correct / total if total > 0 else 0.0

    per_task_metrics = {}
    for task_name, values in per_task.items():
        task_total = values["total"]
        task_correct = values["correct"]
        per_task_metrics[task_name] = {
            "accuracy": task_correct / task_total if task_total > 0 else 0.0,
            "correct": task_correct,
            "total": task_total,
        }

    return {
        "seed": seed,
        "step": step,
        "checkpoint_task": checkpoint_task,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_task": per_task_metrics,
        "examples": rows,
    }


def save_reference_results(
    reference_metrics: dict[str, Any],
    output_dir: str,
    seed: int,
) -> None:
    """Write reference-set predictions, metrics, and cumulative score CSV.

    Args:
        reference_metrics: Metrics dictionary returned by
            ``_evaluate_reference_set``.
        output_dir: Root experiment output directory.
        seed: Experiment seed embedded in the output directory name.
    """
    ref_output_dir = os.path.join(output_dir, f"reference_seed_{seed}")
    os.makedirs(ref_output_dir, exist_ok=True)

    step = reference_metrics["step"]
    checkpoint_task = reference_metrics["checkpoint_task"]

    predictions_path = os.path.join(
        ref_output_dir,
        f"step_{step}_{checkpoint_task}_reference_predictions.jsonl",
    )

    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in reference_metrics.get("examples", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_without_examples = {
        k: v for k, v in reference_metrics.items() if k != "examples"
    }

    metrics_path = os.path.join(
        ref_output_dir,
        f"step_{step}_{checkpoint_task}_reference_metrics.json",
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_without_examples, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(ref_output_dir, "reference_scores.csv")
    file_exists = os.path.exists(summary_path)

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "step",
                "checkpoint_task",
                "reference_accuracy",
                "correct",
                "total",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "seed": reference_metrics["seed"],
                "step": reference_metrics["step"],
                "checkpoint_task": reference_metrics["checkpoint_task"],
                "reference_accuracy": reference_metrics["accuracy"],
                "correct": reference_metrics["correct"],
                "total": reference_metrics["total"],
            }
        )


def evaluate_reference_set(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
    system_prompt: str,
    use_system_prompt: bool,
    step: int,
    checkpoint_task: str,
    seed: int,
    device: torch.device | str,
) -> dict[str, Any] | None:
    """Evaluate and persist metrics for the configured reference set.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer used for prompt rendering and generation.
        config: Experiment configuration.
        system_prompt: Optional system prompt content.
        use_system_prompt: Whether system prompts are included in rendered
            messages.
        step: Continual-learning step for this checkpoint.
        checkpoint_task: Task id or ``"base"`` identifying the checkpoint.
        seed: Experiment seed.
        device: Device used for generation.

    Returns:
        Reference metrics dictionary, or ``None`` when reference-set evaluation
        is disabled.
    """
    if not config.get("reference_set_evaluation", {}).get("enabled", False):
        print(
            "Skipping evaluation on reference set as reference_set_evaluation is disabled in config."
        )
        return None

    print(f"Evaluating reference set at step={step}, checkpoint={checkpoint_task}")

    reference_metrics = _evaluate_reference_set(
        model=model,
        tokenizer=tokenizer,
        config=config,
        system_prompt=system_prompt,
        use_system_prompt=use_system_prompt,
        step=step,
        checkpoint_task=checkpoint_task,
        seed=seed,
        device=device,
    )

    print(
        f"Reference accuracy step={step}, checkpoint={checkpoint_task}: "
        f"{reference_metrics['accuracy']:.4f} "
        f"({reference_metrics['correct']}/{reference_metrics['total']})"
    )

    save_reference_results(
        reference_metrics=reference_metrics,
        output_dir=config["experiment"]["output_dir"],
        seed=seed,
    )

    return reference_metrics
