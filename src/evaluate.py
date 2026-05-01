import re
import os
import csv
import json
import torch

from typing import Dict, Any, List
from datasets import load_dataset

from tqdm.auto import tqdm

from data import build_messages, get_task_prompt
from utils import normalize_answer


def save_eval_results(
    val_metrics: Dict[str, Any],
    output_dir: str,
    task: str,
    seed: int,
    split_name: str = "val",
) -> str:
    eval_output_dir = os.path.join(output_dir, f"eval_{task}_{seed}")
    os.makedirs(eval_output_dir, exist_ok=True)

    predictions_path = os.path.join(eval_output_dir, f"{split_name}_predictions.jsonl")
    metrics_path = os.path.join(eval_output_dir, f"{split_name}_metrics.json")

    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in val_metrics.get("examples", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_without_examples = {
        k: v for k, v in val_metrics.items() if k != "examples"
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_without_examples, f, indent=2, ensure_ascii=False)

    return eval_output_dir


def append_score_rows(
    output_dir: str,
    rows: List[Dict[str, Any]],
    filename: str = "scores.csv",
):
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
    model,
    tokenizer,
    config: Dict[str, Any],
    task_order: List[str],
    task_prompt_map: Dict[str, str],
    system_prompt: str,
    step: int,
    checkpoint_task: str,
    seed: int,
    device,
    split_file_key: str = "val_file",
):
    rows = []

    model.eval()
    model.config.use_cache = True

    for eval_task in task_order:
        task_config = config["tasks"][eval_task]
        task_name = task_config["name"]

        print(
            f"Evaluating step={step}, checkpoint={checkpoint_task}, "
            f"eval_task={eval_task}-{task_name}"
        )

        task_prompt = get_task_prompt(task_name)

        metrics = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            data_file=task_config[split_file_key],
            task_name=task_name,
            task_prompt=task_prompt,
            system_prompt=system_prompt,
            device=device,
            max_examples=config["evaluation"].get("max_eval_examples"),
            max_new_tokens=config["evaluation"]["max_new_tokens"],
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


def generate_answer(
    model,
    tokenizer,
    example: Dict[str, Any],
    task_prompt: str,
    system_prompt: str,
    device,
    max_new_tokens: int,
    enable_thinking: bool = False,
    use_system_prompt: bool = False,
) -> str:
    messages = build_messages(
        prompt=example["prompt"],
        task_prompt=task_prompt,
        system_prompt=system_prompt,
        use_system_prompt=False,
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=False).strip()


def evaluate_accuracy(
    model,
    tokenizer,
    data_file: str,
    task_name: str,
    task_prompt: str,
    system_prompt: str,
    device,
    max_examples: int | None = None,
    max_new_tokens: int = 256,
) -> Dict[str, Any]:
    dataset = load_dataset("json", data_files=data_file, split="train")

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    model.eval()

    correct = 0
    total = 0
    rows: List[Dict[str, Any]] = []

    for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
        gold = normalize_answer(example["answer"], task_name)

        prediction_text = generate_answer(
            model=model,
            tokenizer=tokenizer,
            example=example,
            task_prompt=task_prompt,
            system_prompt=system_prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
        )

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