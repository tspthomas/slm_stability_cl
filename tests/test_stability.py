import csv
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

import stability
from constants import QWEN_MULTIPLE_CHOICE_PROMPT
from stability import (
    _compute_reference_stability_metrics,
    batch_iter,
    build_prompt_texts,
    compute_reference_stability_metrics,
    get_next_token_log_probs,
    save_stability_results,
    summarize,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class FakeBatchEncoding(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakePromptTokenizer:
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
        return "|".join(message["role"] for message in messages)


class FakeLogProbTokenizer:
    def __init__(self):
        self.padding_side = "right"
        self.padding_sides_seen = []

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=None,
        truncation=None,
        max_length=None,
        add_special_tokens=None,
    ):
        self.padding_sides_seen.append(self.padding_side)
        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]]),
                "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
            }
        )


class FakeLogProbModel:
    def __call__(self, **inputs):
        logits = torch.zeros((2, 3, 3), dtype=torch.float32)
        logits[0, 1, :] = torch.tensor([3.0, 0.0, -1.0])
        logits[0, 2, :] = torch.tensor([0.0, 1.0, 2.0])
        logits[1, 2, :] = torch.tensor([2.0, 0.0, -1.0])
        return SimpleNamespace(logits=logits)


class FakeEvalModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=False)
        self.eval_called = False

    def eval(self):
        self.eval_called = True


class FakeAdapterModel(FakeEvalModel):
    def __init__(self):
        super().__init__()
        self.disable_adapter_entered = False

    @contextmanager
    def disable_adapter(self):
        self.disable_adapter_entered = True
        yield


class TestStability(unittest.TestCase):
    def test_batch_iter_yields_fixed_size_batches(self):
        self.assertEqual(
            list(batch_iter(["a", "b", "c", "d", "e"], batch_size=2)),
            [["a", "b"], ["c", "d"], ["e"]],
        )

    def test_build_prompt_texts_renders_task_specific_prompts(self):
        tokenizer = FakePromptTokenizer()
        fixed_kwargs = {"date_string": "26 Jul 2024"}

        texts = build_prompt_texts(
            examples=[{"task_name": " ScienceQA ", "prompt": "Question?"}],
            tokenizer=tokenizer,
            system_prompt="system",
            use_system_prompt=True,
            chat_template_kwargs=fixed_kwargs,
        )

        self.assertEqual(texts, ["system|user"])
        rendered = tokenizer.rendered_messages[0]
        self.assertTrue(rendered["add_generation_prompt"])
        self.assertFalse(rendered["enable_thinking"])
        self.assertEqual(rendered["messages"][0]["role"], "system")
        self.assertIn(QWEN_MULTIPLE_CHOICE_PROMPT, rendered["messages"][1]["content"])
        self.assertEqual(
            rendered["chat_template_kwargs"],
            fixed_kwargs,
        )

    def test_get_next_token_log_probs_selects_last_token_with_left_padding(self):
        tokenizer = FakeLogProbTokenizer()
        model = FakeLogProbModel()

        log_probs = get_next_token_log_probs(
            model=model,
            tokenizer=tokenizer,
            texts=["one", "two"],
            device="cpu",
            max_length=16,
        )

        expected = torch.stack(
            [
                torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=-1),
                torch.log_softmax(torch.tensor([2.0, 0.0, -1.0]), dim=-1),
            ]
        )
        self.assertTrue(torch.allclose(log_probs, expected))
        self.assertEqual(tokenizer.padding_sides_seen, ["left"])
        self.assertEqual(tokenizer.padding_side, "right")

    def test_summarize_returns_mean_and_quantiles(self):
        metrics = summarize([1.0, 2.0, 3.0], "entropy")
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        self.assertEqual(metrics["entropy_mean"], 2.0)
        self.assertEqual(metrics["entropy_p95"], float(torch.quantile(values, 0.95)))
        self.assertEqual(metrics["entropy_p10"], float(torch.quantile(values, 0.10)))

    def test_summarize_returns_nan_for_empty_values(self):
        metrics = summarize([], "margin")

        self.assertTrue(math.isnan(metrics["margin_mean"]))
        self.assertTrue(math.isnan(metrics["margin_p95"]))
        self.assertTrue(math.isnan(metrics["margin_p10"]))

    def test_compute_reference_stability_metrics_internal_computes_entropy_margin_and_kl(self):
        model = FakeAdapterModel()
        dataset = FakeDataset(
            [
                {"task_name": "scienceqa", "prompt": "p1"},
                {"task_name": "fomc", "prompt": "p2"},
            ]
        )
        current_log_probs = torch.log_softmax(
            torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]),
            dim=-1,
        )
        base_log_probs = torch.log_softmax(
            torch.tensor([[1.5, 1.0, 0.5], [0.5, 2.5, 1.0]]),
            dim=-1,
        )

        with (
            patch.object(stability, "load_dataset", return_value=dataset),
            patch.object(stability, "tqdm", side_effect=lambda iterable, **_: iterable),
            patch.object(
                stability,
                "get_next_token_log_probs",
                side_effect=[current_log_probs, base_log_probs],
            ),
        ):
            metrics = _compute_reference_stability_metrics(
                model=model,
                tokenizer=FakePromptTokenizer(),
                reference_file="reference.json",
                system_prompt="system",
                device="cpu",
                seed=33,
                step=1,
                checkpoint_task="task_1",
                batch_size=2,
                max_length=128,
                use_system_prompt=False,
            )

        current_probs = current_log_probs.exp()
        entropy = -(current_probs * current_log_probs).sum(dim=-1)
        margin = torch.topk(current_log_probs, k=2, dim=-1).values
        margin = margin[:, 0] - margin[:, 1]
        kl = (current_probs * (current_log_probs - base_log_probs)).sum(dim=-1)

        self.assertTrue(model.eval_called)
        self.assertTrue(model.config.use_cache)
        self.assertTrue(model.disable_adapter_entered)
        self.assertEqual(metrics["num_reference_examples"], 2)
        self.assertAlmostEqual(metrics["entropy_mean"], float(entropy.mean()))
        self.assertAlmostEqual(metrics["entropy_p95"], float(torch.quantile(entropy, 0.95)))
        self.assertAlmostEqual(metrics["entropy_p10"], float(torch.quantile(entropy, 0.10)))
        self.assertAlmostEqual(metrics["margin_mean"], float(margin.mean()))
        self.assertAlmostEqual(metrics["margin_p95"], float(torch.quantile(margin, 0.95)))
        self.assertAlmostEqual(metrics["margin_p10"], float(torch.quantile(margin, 0.10)))
        self.assertAlmostEqual(metrics["kl_to_base_mean"], float(kl.mean()))
        self.assertAlmostEqual(metrics["kl_to_base_p95"], float(torch.quantile(kl, 0.95)))
        self.assertAlmostEqual(metrics["kl_to_base_p10"], float(torch.quantile(kl, 0.10)))

    def test_compute_reference_stability_metrics_internal_sets_nan_kl_without_adapter(self):
        model = FakeEvalModel()
        dataset = FakeDataset([{"task_name": "scienceqa", "prompt": "p1"}])
        current_log_probs = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.0]]), dim=-1)

        with (
            patch.object(stability, "load_dataset", return_value=dataset),
            patch.object(stability, "tqdm", side_effect=lambda iterable, **_: iterable),
            patch.object(
                stability,
                "get_next_token_log_probs",
                return_value=current_log_probs,
            ),
        ):
            metrics = _compute_reference_stability_metrics(
                model=model,
                tokenizer=FakePromptTokenizer(),
                reference_file="reference.json",
                system_prompt="system",
                device="cpu",
                seed=33,
                step=1,
                checkpoint_task="task_1",
                batch_size=1,
                max_length=128,
                use_system_prompt=False,
            )

        self.assertTrue(math.isnan(metrics["kl_to_base_mean"]))
        self.assertTrue(math.isnan(metrics["kl_to_base_p95"]))
        self.assertTrue(math.isnan(metrics["kl_to_base_p10"]))

    def test_save_stability_results_writes_json_and_flat_csv(self):
        stability_metrics = {
            "seed": 33,
            "step": 1,
            "checkpoint_task": "task_1",
            "entropy_mean": 0.5,
            "margin_mean": 1.5,
            "nested": {"ignored": True},
            "examples": ["ignored"],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            stability_dir = save_stability_results(stability_metrics, tmpdir, seed=33)

            with open(
                os.path.join(stability_dir, "step_1_task_1_stability.json"),
                encoding="utf-8",
            ) as f:
                json_metrics = json.load(f)

            with open(
                os.path.join(stability_dir, "stability_scores.csv"),
                newline="",
                encoding="utf-8",
            ) as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(json_metrics, stability_metrics)
        self.assertEqual(rows[0]["checkpoint_task"], "task_1")
        self.assertNotIn("nested", rows[0])
        self.assertNotIn("examples", rows[0])

    def test_compute_reference_stability_metrics_returns_empty_when_disabled(self):
        with patch("builtins.print"):
            metrics, stability_dir = compute_reference_stability_metrics(
                model=FakeEvalModel(),
                tokenizer=FakePromptTokenizer(),
                config={"stability": {"enabled": False}},
                system_prompt="system",
                use_system_prompt=False,
                device="cpu",
                seed=33,
                step=1,
                checkpoint_task="task_1",
            )

        self.assertEqual(metrics, {})
        self.assertEqual(stability_dir, "")

    def test_compute_reference_stability_metrics_overrides_base_kl_and_saves(self):
        computed_metrics = {
            "seed": 33,
            "step": 0,
            "checkpoint_task": "base",
            "num_reference_examples": 2,
            "entropy_mean": 0.5,
            "margin_mean": 1.0,
            "kl_to_base_mean": math.nan,
            "kl_to_base_p95": math.nan,
            "kl_to_base_p10": math.nan,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "stability": {"enabled": True},
                "reference_set_evaluation": {
                    "file": "reference.json",
                    "batch_size": 2,
                },
                "training": {"max_length": 128},
                "experiment": {"output_dir": tmpdir},
            }

            with (
                patch.object(
                    stability,
                    "_compute_reference_stability_metrics",
                    return_value=computed_metrics,
                ),
                patch("builtins.print"),
            ):
                metrics, stability_dir = compute_reference_stability_metrics(
                    model=FakeEvalModel(),
                    tokenizer=FakePromptTokenizer(),
                    config=config,
                    system_prompt="system",
                    use_system_prompt=False,
                    device="cpu",
                    seed=33,
                    step=0,
                    checkpoint_task="base",
                )

            self.assertEqual(metrics["kl_to_base_mean"], 0.0)
            self.assertEqual(metrics["kl_to_base_p95"], 0.0)
            self.assertEqual(metrics["kl_to_base_p10"], 0.0)
            self.assertTrue(os.path.exists(stability_dir))


if __name__ == "__main__":
    unittest.main()
