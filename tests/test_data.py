import unittest
from types import SimpleNamespace

import torch

from constants import DEFAULT_SYSTEM_PROMPT, QWEN_MULTIPLE_CHOICE_PROMPT
from data import ChatSFTDataset, SFTCollator, build_messages, get_task_prompt


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.map_calls = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def map(self, fn):
        mapped = FakeDataset([{**row, **fn(row)} for row in self.rows])
        mapped.map_calls = self.map_calls + 1
        return mapped


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self):
        self.rendered_messages = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
        **chat_template_kwargs,
    ):
        self.rendered_messages.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "chat_template_kwargs": chat_template_kwargs,
            }
        )

        text = "".join(
            f"<{message['role']}>:{message['content']}\n" for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>:"

        return text

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


class BadPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        if add_generation_prompt:
            return "prompt-prefix"
        return "different-full-text"


class TestData(unittest.TestCase):
    def test_get_task_prompt_normalizes_task_name(self):
        self.assertEqual(get_task_prompt(" ScienceQA "), QWEN_MULTIPLE_CHOICE_PROMPT)

    def test_get_task_prompt_rejects_unknown_task(self):
        with self.assertRaises(KeyError):
            get_task_prompt("unknown")

    def test_build_messages_includes_system_prompt_when_requested(self):
        messages = build_messages(
            prompt="Question?",
            task_prompt="Choose one.",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            use_system_prompt=True,
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Choose one.\n\nQuestion?")

    def test_build_messages_omits_system_prompt_when_disabled(self):
        messages = build_messages(
            prompt=" Question? ",
            task_prompt=" Choose one. ",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            use_system_prompt=False,
        )

        self.assertEqual(
            messages, [{"role": "user", "content": "Choose one.\n\nQuestion?"}]
        )

    def test_chat_sft_dataset_masks_prompt_tokens(self):
        tokenizer = FakeTokenizer()
        dataset = FakeDataset([{"prompt": "Question?", "answer": "A"}])

        sft_dataset = ChatSFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            task_name="fomc",
            max_length=512,
        )
        item = sft_dataset[0]

        prompt_text = tokenizer.apply_chat_template(
            build_messages(
                prompt="Question?",
                task_prompt=get_task_prompt("fomc"),
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                use_system_prompt=False,
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_len = len(prompt_text)

        self.assertEqual(item["input_ids"].dtype, torch.long)
        self.assertEqual(item["attention_mask"].tolist(), [1] * len(item["input_ids"]))
        self.assertEqual(item["labels"][:prompt_len].tolist(), [-100] * prompt_len)
        self.assertNotEqual(item["labels"][prompt_len:].tolist(), [])
        self.assertNotIn(-100, item["labels"][prompt_len:].tolist())

    def test_chat_sft_dataset_preserves_answer_labels_after_left_truncation(self):
        tokenizer = FakeTokenizer()
        dataset = FakeDataset([{"prompt": "Question?", "answer": "ABCDE"}])

        item = ChatSFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            task_name="numgluecm",
            max_length=3,
        )[0]

        self.assertEqual(len(item["input_ids"]), 3)
        self.assertEqual(item["labels"].tolist(), item["input_ids"].tolist())

    def test_chat_sft_dataset_raises_when_prompt_is_not_full_text_prefix(self):
        tokenizer = BadPrefixTokenizer()
        dataset = FakeDataset([{"prompt": "Question?", "answer": "A"}])

        sft_dataset = ChatSFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            task_name="fomc",
        )

        with self.assertRaisesRegex(ValueError, "Prompt tokens are not a prefix"):
            sft_dataset[0]

    def test_chat_sft_dataset_splits_scienceqa_answer_case_insensitively(self):
        tokenizer = FakeTokenizer()
        dataset = FakeDataset(
            [{"prompt": "Question?", "answer": "B\nBecause the evidence supports B."}]
        )

        sft_dataset = ChatSFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            task_name=" ScienceQA ",
        )

        self.assertEqual(sft_dataset.dataset.map_calls, 1)
        self.assertEqual(sft_dataset.dataset[0]["answer"], "B")
        self.assertEqual(
            sft_dataset.dataset[0]["reasoning"],
            "Because the evidence supports B.",
        )

    def test_sft_collator_pads_inputs_labels_and_attention_masks(self):
        tokenizer = FakeTokenizer()
        collator = SFTCollator(tokenizer)

        batch = collator(
            [
                {
                    "input_ids": torch.tensor([1, 2], dtype=torch.long),
                    "labels": torch.tensor([-100, 2], dtype=torch.long),
                    "attention_mask": torch.tensor([1, 1], dtype=torch.long),
                },
                {
                    "input_ids": torch.tensor([3], dtype=torch.long),
                    "labels": torch.tensor([3], dtype=torch.long),
                    "attention_mask": torch.tensor([1], dtype=torch.long),
                },
            ]
        )

        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_sft_collator_falls_back_to_eos_token_for_padding(self):
        tokenizer = FakeTokenizer()
        tokenizer.pad_token_id = None

        collator = SFTCollator(tokenizer)

        self.assertEqual(collator.pad_token_id, tokenizer.eos_token_id)

    def test_sft_collator_requires_pad_or_eos_token(self):
        tokenizer = FakeTokenizer()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = None

        with self.assertRaisesRegex(
            ValueError, "neither pad_token_id nor eos_token_id"
        ):
            SFTCollator(tokenizer)

    def test_chat_sft_dataset_forwards_chat_template_kwargs(self):
        tokenizer = FakeTokenizer()
        dataset = FakeDataset([{"prompt": "Question?", "answer": "A"}])
        fixed_kwargs = {"date_string": "26 Jul 2024"}

        _ = ChatSFTDataset(
            dataset=dataset,
            tokenizer=tokenizer,
            task_name="fomc",
            max_length=512,
            chat_template_kwargs=fixed_kwargs,
        )[0]

        # Training renders the prompt and the complete prompt-answer conversation.
        self.assertEqual(len(tokenizer.rendered_messages), 2)

        for rendered in tokenizer.rendered_messages:
            self.assertEqual(
                rendered["chat_template_kwargs"],
                fixed_kwargs,
            )


if __name__ == "__main__":
    unittest.main()
