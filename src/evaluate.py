import re
import os
import json
import torch

from typing import Dict, Any, List
from datasets import load_dataset

from tqdm.auto import tqdm


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


def build_eval_messages(example: Dict[str, Any], task_prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    user_prompt = task_prompt + example["prompt"].strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def normalize_answer(text: str) -> str:
    text = str(text).strip()

    # Remove common chat/thinking artifacts.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("<|im_end|>", "").strip()

    # Prefer explicit "Final answer: X" if present.
    match = re.search(r"final answer\s*:\s*([A-Za-z0-9\.\-]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().upper()

    # For multiple-choice tasks, pick first A/B/C/D/E if present.
    match = re.search(r"\b([A-E])\b", text.upper())
    if match:
        return match.group(1).strip().upper()

    # Fallback for numeric/string answers.
    return text.splitlines()[0].strip().upper()


def generate_answer(
    model,
    tokenizer,
    example: Dict[str, Any],
    task_prompt: str,
    system_prompt: str,
    device,
    max_new_tokens: int = 16,
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
    max_new_tokens: int = 16,
) -> Dict[str, Any]:
    dataset = load_dataset("json", data_files=data_file, split="train")

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    model.eval()

    correct = 0
    total = 0
    rows: List[Dict[str, Any]] = []

    for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
        gold = normalize_answer(example["answer"])

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

        pred = normalize_answer(prediction_text)

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

        if total == 5:
            break

    accuracy = correct / total if total > 0 else 0.0

    return {
        "task_name": task_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "examples": rows,
    }