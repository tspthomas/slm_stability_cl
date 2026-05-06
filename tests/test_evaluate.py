import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, "./src")
from constants import (
    GENERATION_DO_SAMPLE,
    GENERATION_USE_CACHE,
    QWEN_MULTIPLE_CHOICE_PROMPT,
)
from evaluate import (
    _eos_found,
    batch_iter,
    build_generation_kwargs,
    build_generation_text,
    get_eos_token_ids,
    get_pad_token_id,
    resolve_reference_task_name,
    trim_after_eos,
)


class FakeTokenizer:
    def __init__(
        self,
        pad_token_id=0,
        eos_token_id=99,
        unk_token_id=-1,
        token_ids=None,
    ):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.token_ids = token_ids or {
            "<end_of_turn>": 100,
            "<|im_end|>": 101,
        }
        self.rendered_messages = []

    def convert_tokens_to_ids(self, token):
        return self.token_ids.get(token, self.unk_token_id)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        self.rendered_messages.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        return "|".join(message["role"] for message in messages)


class TestEvaluateHelpers(unittest.TestCase):
    def test_batch_iter_yields_fixed_size_batches(self):
        self.assertEqual(
            list(batch_iter(["a", "b", "c", "d", "e"], batch_size=2)),
            [["a", "b"], ["c", "d"], ["e"]],
        )

    def test_eos_found_supports_none_single_and_multiple_ids(self):
        generated_ids = torch.tensor([4, 5, 6])

        self.assertFalse(_eos_found(generated_ids, None))
        self.assertTrue(_eos_found(generated_ids, 5))
        self.assertTrue(_eos_found(generated_ids, [9, 6]))
        self.assertFalse(_eos_found(generated_ids, [9, 10]))

    def test_build_generation_text_renders_chat_template(self):
        tokenizer = FakeTokenizer()

        text = build_generation_text(
            tokenizer=tokenizer,
            example={"prompt": "Question?"},
            task_prompt=QWEN_MULTIPLE_CHOICE_PROMPT,
            system_prompt="system prompt",
            use_system_prompt=True,
            enable_thinking=True,
        )

        self.assertEqual(text, "system|user")
        self.assertEqual(tokenizer.rendered_messages[0]["add_generation_prompt"], True)
        self.assertEqual(tokenizer.rendered_messages[0]["enable_thinking"], True)
        self.assertEqual(tokenizer.rendered_messages[0]["messages"][0]["role"], "system")
        self.assertEqual(tokenizer.rendered_messages[0]["messages"][1]["role"], "user")

    def test_get_pad_token_id_prefers_pad_and_falls_back_to_eos(self):
        self.assertEqual(get_pad_token_id(FakeTokenizer(pad_token_id=7)), 7)
        self.assertEqual(
            get_pad_token_id(FakeTokenizer(pad_token_id=None, eos_token_id=9)),
            9,
        )

    def test_get_pad_token_id_requires_pad_or_eos(self):
        with self.assertRaisesRegex(ValueError, "neither pad_token_id nor eos_token_id"):
            get_pad_token_id(FakeTokenizer(pad_token_id=None, eos_token_id=None))

    def test_get_eos_token_ids_collects_unique_tokenizer_model_and_chat_ids(self):
        tokenizer = FakeTokenizer(eos_token_id=99)
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[99, 102]))

        self.assertEqual(get_eos_token_ids(tokenizer, model), [99, 102, 100, 101])

    def test_get_eos_token_ids_returns_none_when_no_ids_are_available(self):
        tokenizer = FakeTokenizer(
            eos_token_id=None,
            token_ids={"<end_of_turn>": -1, "<|im_end|>": -1},
        )

        self.assertIsNone(get_eos_token_ids(tokenizer))

    def test_trim_after_eos_keeps_first_eos_token(self):
        generated_ids = torch.tensor([1, 2, 99, 3, 100])

        self.assertEqual(trim_after_eos(generated_ids, 99).tolist(), [1, 2, 99])
        self.assertEqual(trim_after_eos(generated_ids, [100, 99]).tolist(), [1, 2, 99])
        self.assertIs(trim_after_eos(generated_ids, None), generated_ids)
        self.assertIs(trim_after_eos(generated_ids, 200), generated_ids)

    def test_build_generation_kwargs_uses_defaults_without_sampling_extras(self):
        kwargs = build_generation_kwargs({})

        self.assertEqual(kwargs["do_sample"], GENERATION_DO_SAMPLE)
        self.assertEqual(kwargs["use_cache"], GENERATION_USE_CACHE)
        self.assertEqual(kwargs["num_beams"], 1)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)

    def test_build_generation_kwargs_includes_sampling_options_when_enabled(self):
        kwargs = build_generation_kwargs(
            {
                "generation": {
                    "do_sample": True,
                    "use_cache": False,
                    "num_beams": 3,
                    "temperature": 0.7,
                    "top_p": 0.8,
                }
            }
        )

        self.assertEqual(
            kwargs,
            {
                "do_sample": True,
                "use_cache": False,
                "num_beams": 3,
                "temperature": 0.7,
                "top_p": 0.8,
            },
        )

    def test_resolve_reference_task_name_prefers_explicit_task_name(self):
        self.assertEqual(
            resolve_reference_task_name(
                {"task_name": " ScienceQA ", "task_id": "task_1"},
                {"tasks": {"task_1": {"name": "numgluecm"}}},
            ),
            "scienceqa",
        )

    def test_resolve_reference_task_name_falls_back_to_task_id(self):
        self.assertEqual(
            resolve_reference_task_name(
                {"task_id": "task_1"},
                {"tasks": {"task_1": {"name": " FOMC "}}},
            ),
            "fomc",
        )

    def test_resolve_reference_task_name_requires_name_or_known_id(self):
        with self.assertRaisesRegex(ValueError, "Reference example must contain"):
            resolve_reference_task_name({"task_id": "missing"}, {"tasks": {}})


if __name__ == "__main__":
    unittest.main()
