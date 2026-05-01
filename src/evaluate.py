import re
import os
import csv
import json
import torch

from typing import Dict, Any, List
from datasets import load_dataset

from tqdm.auto import tqdm

from constants import MULTIPLE_CHOICE_TASKS, NUMERIC_TASKS


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

    for eval_task in task_order:
        task_config = config["tasks"][eval_task]
        task_name = task_config["name"]

        print(
            f"Evaluating step={step}, checkpoint={checkpoint_task}, "
            f"eval_task={eval_task}-{task_name}"
        )

        metrics = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            data_file=task_config[split_file_key],
            task_name=task_name,
            task_prompt=task_prompt_map.get(task_name, ""),
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


def build_eval_messages(example: Dict[str, Any], task_prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    user_prompt = example["prompt"].strip() + "\n\n" + task_prompt.strip()

    return [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def strip_generation_artifacts(text: str) -> str:
    text = str(text).strip()

    # Remove thinking blocks and common chat markers.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("<|im_end|>", "").strip()
    text = text.replace("<|endoftext|>", "").strip()

    # Remove markdown emphasis.
    text = text.replace("**", "").replace("__", "")

    return text.strip()


def normalize_number(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")

    # Remove trailing punctuation.
    text = re.sub(r"[.$,;:]+$", "", text)

    # Normalize 186.0 -> 186
    try:
        value = float(text)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return text


def extract_number(text: str) -> str:
    text = strip_generation_artifacts(text)

    # Prefer text after common final-answer markers.
    marker_patterns = [
        r"final answer\s*[:\-]\s*(.*)",
        r"answer\s*[:\-]\s*(.*)",
        r"the answer is\s*(.*)",
        r"therefore,?\s*(.*)",
    ]

    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1)
            numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", candidate)
            if numbers:
                return normalize_number(numbers[-1])

    # Fallback: use the last number in the whole generation.
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if numbers:
        return normalize_number(numbers[-1])

    return text.splitlines()[-1].strip().upper() if text.splitlines() else text.upper()


def extract_multiple_choice(text: str) -> str:
    text = strip_generation_artifacts(text)

    # Prefer explicit answer markers.
    marker_patterns = [
        r"final answer\s*[:\-]\s*([A-E])\b",
        r"answer\s*[:\-]\s*([A-E])\b",
        r"the answer is\s*([A-E])\b",
        r"option\s*([A-E])\b",
        r"choice\s*([A-E])\b",
    ]

    for pattern in marker_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Common case: model outputs just "A" or "A. something".
    match = re.match(r"^\s*([A-E])(?:[\.\)]|\s|$)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: choose the last standalone option, not the first.
    # This avoids picking A from a restated prompt before the final answer.
    matches = re.findall(r"\b([A-E])\b", text.upper())
    if matches:
        return matches[-1]

    return text.splitlines()[-1].strip().upper() if text.splitlines() else text.upper()


def normalize_answer(text: str, task_name: str = None) -> str:
    text = strip_generation_artifacts(text)

    if task_name in MULTIPLE_CHOICE_TASKS:
        return extract_multiple_choice(text)

    if task_name in NUMERIC_TASKS:
        return extract_number(text)

    # Generic fallback.
    # First try final-answer markers.
    match = re.search(
        r"final answer\s*[:\-]\s*(.+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).splitlines()[0].strip().upper()

    return text.splitlines()[-1].strip().upper() if text.splitlines() else text.upper()


def generate_answer(
    model,
    tokenizer,
    example: Dict[str, Any],
    task_prompt: str,
    system_prompt: str,
    device,
    max_new_tokens: int,
    enable_thinking: bool = False,
) -> str:
    messages = build_eval_messages(example, task_prompt, system_prompt)

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