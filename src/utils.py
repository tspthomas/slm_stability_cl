import re
import torch
import numpy as np

from constants import MULTIPLE_CHOICE_TASKS, NUMERIC_TASKS


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """
    Determine the available device (GPU, MPS, or CPU) for computation.

    Returns:
        str: The name of the device to use ("cuda", "mps", or "cpu").
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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