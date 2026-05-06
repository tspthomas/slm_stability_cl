"""Shared utility functions for reproducible experiments and answer parsing.

This module contains small helpers used across training and evaluation: random
seed setup, device selection, LoRA configuration detection, and task-aware
normalization of generated answers.
"""

import re

import numpy as np
import torch

from constants import MULTIPLE_CHOICE_TASKS, NUMERIC_TASKS


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible NumPy and PyTorch behavior.

    Args:
        seed: Seed value passed to the supported random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return the best available compute device name.

    Returns:
        One of ``"mps"``, ``"cuda"``, or ``"cpu"``. MPS is preferred on
        supported Apple hardware, followed by CUDA, then CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_config_lora(config: dict[str, object]) -> bool:
    """Return whether an experiment config requests LoRA fine-tuning.

    Args:
        config: Experiment configuration dictionary.

    Returns:
        ``True`` when ``config["peft"]["method"]`` is ``"lora"``.
    """
    is_lora = config.get("peft", {}) is not None
    return is_lora and config.get("peft", {}).get("method", "") == "lora"


def strip_generation_artifacts(text: str) -> str:
    """Remove common model-generation artifacts from answer text.

    This strips thinking blocks, chat-template end markers, and markdown
    emphasis before downstream task-specific parsing.

    Args:
        text: Generated model text or reference answer text.

    Returns:
        Cleaned text with surrounding whitespace removed.
    """
    text = str(text).strip()

    # Remove thinking blocks and common chat markers.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = text.replace("<|im_end|>", "").strip()
    text = text.replace("<|endoftext|>", "").strip()

    # Remove markdown emphasis.
    text = text.replace("**", "").replace("__", "")

    return text.strip()


def normalize_number(text: str) -> str:
    """Normalize a numeric string for exact-match comparison.

    Commas and trailing punctuation are removed. Integer-valued floats are
    canonicalized to integer strings, e.g. ``"186.0"`` becomes ``"186"``.

    Args:
        text: Numeric candidate text.

    Returns:
        Normalized numeric string, or the original cleaned text when parsing as
        a float fails.
    """
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
    """Extract the most likely final numeric answer from generated text.

    Explicit final-answer markers are preferred. If no marker contains a
    number, the last number in the whole generation is returned.

    Args:
        text: Generated model text or reference answer text.

    Returns:
        Extracted and normalized number, or a cleaned fallback string when no
        number is present.
    """
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
    """Extract the most likely final multiple-choice option.

    The parser supports answer markers such as ``"final answer: C"`` and falls
    back to the final standalone option letter from A through E.

    Args:
        text: Generated model text or reference answer text.

    Returns:
        Uppercase option letter, or a cleaned fallback string when no option is
        found.
    """
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


def normalize_answer(text: str, task_name: str | None = None) -> str:
    """Normalize an answer according to the task's expected answer format.

    Multiple-choice tasks are reduced to an option letter, numeric tasks are
    reduced to a canonical number string, and unknown tasks use a generic
    final-answer fallback.

    Args:
        text: Generated model text or reference answer text.
        task_name: Task identifier used to select a normalization strategy.

    Returns:
        Normalized answer string for exact-match evaluation.
    """
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
