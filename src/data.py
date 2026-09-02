"""Dataset and collation utilities for chat-style supervised fine-tuning.

This module converts prompt/answer examples into chat-template training
examples for causal language model SFT. It also provides a padding collator that
masks prompt tokens with ``-100`` so the loss is computed only on assistant
answer tokens.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset

from constants import DEFAULT_SYSTEM_PROMPT, TASK_PROMPT_MAP, TASK_TRACE_SCIENCEQA


def _split_answer(example: dict[str, Any]) -> dict[str, str]:
    """Split a ScienceQA answer into final answer and reasoning fields.

    ScienceQA examples store the option letter and explanatory rationale in the
    same ``answer`` field. Training uses only the option letter as the assistant
    target, while the rationale is kept separately for downstream inspection.

    Args:
        example: Dataset row with an ``answer`` value containing a newline.

    Returns:
        A partial row update with ``answer`` and ``reasoning`` fields.
    """
    answer, last = example["answer"].split("\n", 1)
    return {"answer": answer, "reasoning": last}


def get_task_prompt(task_name: str) -> str:
    """Return the instruction prompt associated with a task name.

    Args:
        task_name: Task identifier, such as ``"scienceqa"`` or
            ``"numgluecm"``. Leading/trailing whitespace and casing are
            normalized.

    Returns:
        Prompt string used before each example prompt.

    Raises:
        KeyError: If no prompt is configured for the normalized task name.
    """
    task_name = task_name.strip().lower()

    if task_name not in TASK_PROMPT_MAP:
        raise KeyError(
            f"No task prompt found for task_name='{task_name}'. "
            f"Available task prompts: {list(TASK_PROMPT_MAP.keys())}"
        )

    return TASK_PROMPT_MAP[task_name]


def render_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool = False,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render messages using reproducible model-specific template options.

    Args:
        tokenizer: Hugging Face tokenizer providing ``apply_chat_template``.
        messages: Chat messages to render.
        add_generation_prompt: Whether to append the assistant prompt marker.
        enable_thinking: Whether supported templates should enable reasoning.
        chat_template_kwargs: Optional model-specific values passed to the template,
            such as a fixed Llama ``date_string``.

    Returns:
        Rendered chat prompt.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        **dict(chat_template_kwargs or {}),
    )


def build_messages(
    prompt: str,
    task_prompt: str,
    system_prompt: str | None = None,
    use_system_prompt: bool = True,
) -> list[dict[str, str]]:
    """Build chat messages for one user prompt.

    Args:
        prompt: Raw task example prompt.
        task_prompt: Task-level instruction prepended to the raw prompt.
        system_prompt: Optional system message content.
        use_system_prompt: Whether to include ``system_prompt`` when it is
            provided.

    Returns:
        Chat-template-compatible message dictionaries.
    """
    user_text = f"{task_prompt.strip()}\n\n{prompt.strip()}"

    messages: list[dict[str, str]] = []

    if use_system_prompt and system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_text})

    return messages


class ChatSFTDataset(TorchDataset):
    """Torch dataset that renders prompt/answer rows as SFT token tensors."""

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        task_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        use_system_prompt: bool = False,
        max_length: int = 512,
        enable_thinking: bool = False,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a chat-format SFT dataset.

        Args:
            dataset: Row-indexable dataset with ``prompt`` and ``answer``
                fields and, for ScienceQA preprocessing, a ``map`` method.
            tokenizer: Tokenizer implementing ``apply_chat_template`` and
                callable tokenization.
            task_name: Task identifier used to choose the task prompt.
            system_prompt: Optional system message text.
            use_system_prompt: Whether to include the system message in each
                rendered prompt.
            max_length: Maximum number of tokens retained from the rendered
                full conversation. Long examples are left-truncated.
            enable_thinking: Passed through to tokenizer chat-template
                rendering for models that support thinking blocks.
            chat_template_args: Optional model-specific values passed to the template,
                such as a fixed Llama ``date_string``.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_name = task_name.strip().lower()
        self.task_prompt = get_task_prompt(self.task_name)
        self.system_prompt = system_prompt
        self.use_system_prompt = use_system_prompt
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

        # Special handling for ScienceQA to split answer and reasoning for evaluation purposes.
        if self.task_name == TASK_TRACE_SCIENCEQA:
            self.dataset = self.dataset.map(_split_answer)

    def __len__(self) -> int:
        """Return the number of examples in the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Tokenize one example as model inputs, labels, and attention mask.

        Prompt tokens are masked with ``-100`` in ``labels`` so only the
        assistant answer contributes to the SFT loss.

        Args:
            idx: Example index.

        Returns:
            A dictionary containing ``input_ids``, ``labels``, and
            ``attention_mask`` tensors.

        Raises:
            ValueError: If prompt tokens are not a prefix of the full
                conversation tokens, or truncation removes all answer labels.
        """
        ex = self.dataset[int(idx)]

        answer_text = str(ex["answer"]).strip()

        prompt_messages = build_messages(
            prompt=ex["prompt"],
            task_prompt=self.task_prompt,
            system_prompt=self.system_prompt,
            use_system_prompt=self.use_system_prompt,
        )

        full_messages = prompt_messages + [
            {"role": "assistant", "content": answer_text}
        ]

        prompt_text = render_chat_template(
            tokenizer=self.tokenizer,
            messages=prompt_messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        full_text = render_chat_template(
            tokenizer=self.tokenizer,
            messages=full_messages,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
            chat_template_kwargs=self.chat_template_kwargs,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        ).input_ids

        input_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
        ).input_ids

        # sanity check
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Prompt tokens are not a prefix of full conversation tokens. "
                "The chat template may be inconsistent between prompt_text and full_text."
            )

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]

        was_truncated = len(input_ids) > self.max_length
        if was_truncated:
            input_ids = input_ids[-self.max_length :]
            labels = labels[-self.max_length :]

        if all(label == -100 for label in labels):
            raise ValueError(
                "All labels are masked after truncation. "
                "Increase max_length or check the prompt/answer format."
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class SFTCollator:
    """Pad SFT examples into a batch suitable for causal LM training."""

    def __init__(self, tokenizer: Any) -> None:
        """Initialize the collator with tokenizer padding metadata.

        Args:
            tokenizer: Tokenizer with ``pad_token_id`` and/or ``eos_token_id``.

        Raises:
            ValueError: If neither padding nor EOS token ids are available.
        """
        self.tokenizer = tokenizer

        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id

        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id nor eos_token_id. "
                "Please define a pad token before training."
            )

    def __call__(
        self,
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Pad a list of variable-length examples into one batch.

        Args:
            batch: Examples returned by ``ChatSFTDataset``.

        Returns:
            Batched ``input_ids``, ``labels``, and ``attention_mask`` tensors.
        """
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch],
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch],
            batch_first=True,
            padding_value=0,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
