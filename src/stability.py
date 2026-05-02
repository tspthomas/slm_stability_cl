"""
This module implements the stability evaluation logic.
"""

import csv
import json
import math
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm.auto import tqdm

from data import build_messages, get_task_prompt


def batch_iter(dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        yield [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]


def build_prompt_texts(
    examples: List[Dict[str, Any]],
    tokenizer,
    system_prompt: str,
    use_system_prompt: bool,
) -> List[str]:
    texts = []

    for ex in examples:
        task_name = ex["task_name"].strip().lower()
        task_prompt = get_task_prompt(task_name)

        messages = build_messages(
            prompt=ex["prompt"],
            task_prompt=task_prompt,
            system_prompt=system_prompt,
            use_system_prompt=use_system_prompt,
        )

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        texts.append(text)

    return texts


def get_next_token_log_probs(
    model, tokenizer, texts: List[str], device, max_length: int
):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    ).to(device)

    tokenizer.padding_side = old_padding_side

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits  # [batch, seq_len, vocab]

    # Last non-padding token for each example
    attention_mask = inputs["attention_mask"]
    last_indices = attention_mask.sum(dim=1) - 1

    batch_indices = torch.arange(logits.size(0), device=logits.device)
    next_token_logits = logits[batch_indices, last_indices, :].float()

    return F.log_softmax(next_token_logits, dim=-1)


def summarize(values: List[float], name: str) -> Dict[str, float]:
    if not values:
        return {
            f"{name}_mean": math.nan,
            f"{name}_p95": math.nan,
            f"{name}_p10": math.nan,
        }

    x = torch.tensor(values, dtype=torch.float32)

    return {
        f"{name}_mean": float(x.mean().item()),
        f"{name}_p95": float(torch.quantile(x, 0.95).item()),
        f"{name}_p10": float(torch.quantile(x, 0.10).item()),
    }


def _compute_reference_stability_metrics(
    model,
    tokenizer,
    reference_file: str,
    system_prompt: str,
    device,
    seed: int,
    step: int,
    checkpoint_task: str,
    batch_size: int = 4,
    max_length: int = 512,
    use_system_prompt: bool = True,
) -> Dict[str, Any]:
    dataset = load_dataset("json", data_files=reference_file, split="train")

    model.eval()
    model.config.use_cache = True

    entropy_values = []
    margin_values = []
    kl_values = []

    for examples in tqdm(
        batch_iter(dataset, batch_size),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc=f"Stability step={step}",
    ):
        texts = build_prompt_texts(
            examples=examples,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            use_system_prompt=use_system_prompt,
        )

        current_log_probs = get_next_token_log_probs(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=max_length,
        )

        current_probs = current_log_probs.exp()

        entropy = -(current_probs * current_log_probs).sum(dim=-1)

        top2 = torch.topk(current_log_probs, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

        entropy_values.extend(entropy.cpu().tolist())
        margin_values.extend(margin.cpu().tolist())

        # LoRA case: compare current adapter model against base model without adapter.
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                base_log_probs = get_next_token_log_probs(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    device=device,
                    max_length=max_length,
                )

            # KL(current || base), same direction as your old implementation.
            kl = (current_probs * (current_log_probs - base_log_probs)).sum(dim=-1)
            kl_values.extend(kl.cpu().tolist())

    metrics = {
        "seed": seed,
        "step": step,
        "checkpoint_task": checkpoint_task,
        "num_reference_examples": len(dataset),
    }

    metrics.update(summarize(entropy_values, "entropy"))
    metrics.update(summarize(margin_values, "margin"))

    if kl_values:
        metrics.update(summarize(kl_values, "kl_to_base"))
    else:
        metrics["kl_to_base_mean"] = math.nan
        metrics["kl_to_base_p95"] = math.nan
        metrics["kl_to_base_p10"] = math.nan

    return metrics


def save_stability_results(
    stability_metrics: Dict[str, Any],
    output_dir: str,
    seed: int,
    filename: str = "stability_scores.csv",
) -> str:
    """
    Save stability metrics for one checkpoint.

    Writes:
      1. One JSON file per checkpoint with the full metrics dict.
      2. One cumulative CSV file with one row per checkpoint.

    Example output:
      runs/my_run/stability_seed_33/
        step_0_base_stability.json
        step_1_task_1_stability.json
        step_2_task_2_stability.json
        stability_scores.csv
    """
    stability_dir = os.path.join(output_dir, f"stability_seed_{seed}")
    os.makedirs(stability_dir, exist_ok=True)

    step = stability_metrics["step"]
    checkpoint_task = stability_metrics["checkpoint_task"]

    # Save full JSON for this checkpoint.
    json_path = os.path.join(
        stability_dir,
        f"step_{step}_{checkpoint_task}_stability.json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stability_metrics, f, indent=2, ensure_ascii=False)

    # Append flat row to cumulative CSV.
    csv_path = os.path.join(stability_dir, filename)

    flat_metrics = {
        key: value
        for key, value in stability_metrics.items()
        if not isinstance(value, (dict, list))
    }

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_metrics.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(flat_metrics)

    return stability_dir


def compute_reference_stability_metrics(
    model,
    tokenizer,
    config: Dict[str, Any],
    system_prompt: str,
    use_system_prompt: bool,
    device,
    seed: int,
    step: int,
    checkpoint_task: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Compute reference stability metrics for one checkpoint and save them.

    Returns:
        stability_metrics: computed metrics dict
        stability_dir: directory where results were saved
    """
    if not config.get("stability", {}).get("enabled", True):
        print("Stability metrics disabled. Skipping.")
        return {}, ""

    print(
        f"Computing stability metrics | "
        f"seed={seed}, step={step}, checkpoint={checkpoint_task}"
    )

    stability_metrics = _compute_reference_stability_metrics(
        model=model,
        tokenizer=tokenizer,
        reference_file=config["reference_set_evaluation"]["file"],
        system_prompt=system_prompt,
        use_system_prompt=use_system_prompt,
        device=device,
        seed=seed,
        step=step,
        checkpoint_task=checkpoint_task,
        batch_size=config["reference_set_evaluation"].get("batch_size", 4),
        max_length=config["training"]["max_length"],
    )

    # For the base checkpoint, we don't have a "base" model to compare against, so we set KL to base metrics to 0.
    if step == 0 and checkpoint_task == "base":
        stability_metrics["kl_to_base_mean"] = 0.0
        stability_metrics["kl_to_base_p95"] = 0.0
        stability_metrics["kl_to_base_p10"] = 0.0

    stability_dir = save_stability_results(
        stability_metrics=stability_metrics,
        output_dir=config["experiment"]["output_dir"],
        seed=seed,
    )

    print(
        f"Stability metrics saved | "
        f"entropy_mean={stability_metrics.get('entropy_mean'):.4f}, "
        f"margin_mean={stability_metrics.get('margin_mean'):.4f}, "
        f"kl_to_base_mean={stability_metrics.get('kl_to_base_mean'):.4f}"
    )

    return stability_metrics, stability_dir
