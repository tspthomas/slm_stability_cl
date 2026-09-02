import csv
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import evaluate
from constants import (
    GENERATION_DO_SAMPLE,
    GENERATION_USE_CACHE,
    QWEN_MULTIPLE_CHOICE_PROMPT,
)
from evaluate import (
    _eos_found,
    _evaluate_reference_set,
    append_score_rows,
    batch_iter,
    build_generation_kwargs,
    build_generation_text,
    evaluate_accuracy,
    evaluate_checkpoint_on_all_tasks,
    evaluate_reference_set,
    generate_batch_from_texts,
    get_eos_token_ids,
    get_pad_token_id,
    resolve_reference_task_name,
    save_eval_results,
    save_reference_results,
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


class FakeBatchEncoding(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeGenerationTokenizer(FakeTokenizer):
    def __init__(self):
        super().__init__(pad_token_id=None, eos_token_id=99)
        self.eos_token = "<eos>"
        self.pad_token = None
        self.padding_side = "right"
        self.tokenized_padding_sides = []

    def __call__(
        self, texts, return_tensors=None, padding=None, add_special_tokens=None
    ):
        self.tokenized_padding_sides.append(self.padding_side)
        input_ids = torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
            ],
            dtype=torch.long,
        )
        return FakeBatchEncoding({"input_ids": input_ids})

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids.tolist())


class FakeGenerationModel:
    def __init__(self):
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor(
            [
                [10, 11, 12, 7, 99, 8],
                [20, 21, 22, 7, 8, 9],
            ],
            dtype=torch.long,
        )


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def select(self, indices):
        return FakeDataset([self.rows[idx] for idx in indices])


class FakeEvalModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=False)
        self.eval_called = False

    def eval(self):
        self.eval_called = True


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
        fixed_kwargs = {"date_string": "26 Jul 2024"}

        text = build_generation_text(
            tokenizer=tokenizer,
            example={"prompt": "Question?"},
            task_prompt=QWEN_MULTIPLE_CHOICE_PROMPT,
            system_prompt="system prompt",
            use_system_prompt=True,
            enable_thinking=True,
            chat_template_kwargs=fixed_kwargs,
        )

        self.assertEqual(text, "system|user")
        self.assertEqual(tokenizer.rendered_messages[0]["add_generation_prompt"], True)
        self.assertEqual(tokenizer.rendered_messages[0]["enable_thinking"], True)
        self.assertEqual(
            tokenizer.rendered_messages[0]["messages"][0]["role"], "system"
        )
        self.assertEqual(tokenizer.rendered_messages[0]["messages"][1]["role"], "user")
        self.assertEqual(
            tokenizer.rendered_messages[0]["chat_template_kwargs"],
            fixed_kwargs,
        )

    def test_get_pad_token_id_prefers_pad_and_falls_back_to_eos(self):
        self.assertEqual(get_pad_token_id(FakeTokenizer(pad_token_id=7)), 7)
        self.assertEqual(
            get_pad_token_id(FakeTokenizer(pad_token_id=None, eos_token_id=9)),
            9,
        )

    def test_get_pad_token_id_requires_pad_or_eos(self):
        with self.assertRaisesRegex(
            ValueError, "neither pad_token_id nor eos_token_id"
        ):
            get_pad_token_id(FakeTokenizer(pad_token_id=None, eos_token_id=None))

    def test_get_eos_token_ids_collects_unique_tokenizer_model_and_chat_ids(self):
        tokenizer = FakeTokenizer(eos_token_id=99)
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[99, 102])
        )

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


class TestEvaluateWorkflows(unittest.TestCase):
    def test_generate_batch_from_texts_restores_padding_and_reports_max_tokens(self):
        tokenizer = FakeGenerationTokenizer()
        model = FakeGenerationModel()

        results = generate_batch_from_texts(
            model=model,
            tokenizer=tokenizer,
            config={},
            texts=["one", "two"],
            device="cpu",
            max_new_tokens=3,
        )

        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(tokenizer.tokenized_padding_sides, ["left"])
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(results, [("7 99", False), ("7 8 9", True)])
        self.assertEqual(model.generate_kwargs["pad_token_id"], 99)
        self.assertEqual(model.generate_kwargs["max_new_tokens"], 3)

    def test_save_eval_results_writes_predictions_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = save_eval_results(
                val_metrics={
                    "task_name": "scienceqa",
                    "accuracy": 0.5,
                    "examples": [{"prompt": "p1"}, {"prompt": "p2"}],
                },
                output_dir=tmpdir,
                task="task_1",
                seed=33,
                split_name="val",
            )

            with open(
                os.path.join(output_dir, "val_predictions.jsonl"),
                encoding="utf-8",
            ) as f:
                prediction_rows = [json.loads(line) for line in f]

            with open(
                os.path.join(output_dir, "val_metrics.json"),
                encoding="utf-8",
            ) as f:
                metrics = json.load(f)

        self.assertEqual(prediction_rows, [{"prompt": "p1"}, {"prompt": "p2"}])
        self.assertEqual(metrics, {"task_name": "scienceqa", "accuracy": 0.5})

    def test_append_score_rows_writes_header_once(self):
        rows = [
            {
                "seed": 1,
                "step": 0,
                "checkpoint_task": "base",
                "eval_task": "task_1",
                "eval_task_name": "scienceqa",
                "accuracy": 1.0,
                "correct": 2,
                "total": 2,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            append_score_rows(tmpdir, rows, filename="scores.csv")
            append_score_rows(tmpdir, rows, filename="scores.csv")

            with open(
                os.path.join(tmpdir, "scores.csv"), newline="", encoding="utf-8"
            ) as f:
                csv_rows = list(csv.DictReader(f))

        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(csv_rows[0]["checkpoint_task"], "base")
        self.assertEqual(csv_rows[1]["accuracy"], "1.0")

    def test_evaluate_checkpoint_on_all_tasks_returns_none_when_disabled(self):
        model = FakeEvalModel()

        with patch("builtins.print"):
            rows = evaluate_checkpoint_on_all_tasks(
                model=model,
                tokenizer=object(),
                config={"eval_set_evaluation": {"enabled": False}},
                task_order=["task_1"],
                system_prompt="system",
                use_system_prompt=False,
                step=0,
                checkpoint_task="base",
                seed=1,
                device="cpu",
            )

        self.assertIsNone(rows)
        self.assertFalse(model.eval_called)

    def test_evaluate_checkpoint_on_all_tasks_saves_rows_when_enabled(self):
        model = FakeEvalModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "eval_set_evaluation": {
                    "enabled": True,
                    "split_file_key": "val_file",
                    "max_examples": 2,
                    "max_new_tokens": 4,
                },
                "experiment": {"output_dir": tmpdir},
                "tasks": {"task_1": {"name": "scienceqa", "val_file": "val.json"}},
            }

            with (
                patch.object(
                    evaluate,
                    "evaluate_accuracy",
                    return_value={
                        "accuracy": 1.0,
                        "correct": 2,
                        "total": 2,
                        "examples": [{"prompt": "p"}],
                    },
                ) as eval_accuracy,
                patch("builtins.print"),
            ):
                rows = evaluate_checkpoint_on_all_tasks(
                    model=model,
                    tokenizer=object(),
                    config=config,
                    task_order=["task_1"],
                    system_prompt="system",
                    use_system_prompt=True,
                    step=1,
                    checkpoint_task="task_1",
                    seed=33,
                    device="cpu",
                )

            self.assertEqual(rows[0]["accuracy"], 1.0)
            self.assertTrue(model.eval_called)
            self.assertTrue(model.config.use_cache)
            eval_accuracy.assert_called_once()
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "scores_seed_33.csv")))

    def test_evaluate_accuracy_computes_exact_match_metrics(self):
        model = FakeEvalModel()
        dataset = FakeDataset(
            [
                {"prompt": "p1", "answer": "A"},
                {"prompt": "p2", "answer": "B"},
            ]
        )

        with (
            patch.object(evaluate, "load_dataset", return_value=dataset),
            patch.object(evaluate, "tqdm", side_effect=lambda iterable, **_: iterable),
            patch.object(
                evaluate,
                "generate_batch_from_texts",
                return_value=[("A", False), ("C", True)],
            ),
        ):
            metrics = evaluate_accuracy(
                model=model,
                tokenizer=FakeTokenizer(),
                config={},
                data_file="data.json",
                task_name="scienceqa",
                task_prompt=QWEN_MULTIPLE_CHOICE_PROMPT,
                system_prompt="system",
                use_system_prompt=False,
                device="cpu",
                batch_size=2,
            )

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["examples"][1]["prediction"], "C")
        self.assertTrue(metrics["examples"][1]["hit_max_tokens"])

    def test_evaluate_accuracy_applies_max_examples(self):
        model = FakeEvalModel()
        dataset = FakeDataset(
            [
                {"prompt": "p1", "answer": "A"},
                {"prompt": "p2", "answer": "B"},
            ]
        )

        with (
            patch.object(evaluate, "load_dataset", return_value=dataset),
            patch.object(evaluate, "tqdm", side_effect=lambda iterable, **_: iterable),
            patch.object(
                evaluate,
                "generate_batch_from_texts",
                return_value=[("A", False)],
            ),
            patch("builtins.print"),
        ):
            metrics = evaluate_accuracy(
                model=model,
                tokenizer=FakeTokenizer(),
                config={},
                data_file="data.json",
                task_name="scienceqa",
                task_prompt=QWEN_MULTIPLE_CHOICE_PROMPT,
                system_prompt="system",
                use_system_prompt=False,
                device="cpu",
                max_examples=1,
                batch_size=2,
            )

        self.assertEqual(metrics["total"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_evaluate_reference_set_returns_none_when_disabled(self):
        with patch("builtins.print"):
            metrics = evaluate_reference_set(
                model=FakeEvalModel(),
                tokenizer=object(),
                config={"reference_set_evaluation": {"enabled": False}},
                system_prompt="system",
                use_system_prompt=False,
                step=0,
                checkpoint_task="base",
                seed=1,
                device="cpu",
            )

        self.assertIsNone(metrics)

    def test_evaluate_reference_set_saves_metrics_when_enabled(self):
        reference_metrics = {
            "seed": 33,
            "step": 1,
            "checkpoint_task": "task_1",
            "accuracy": 1.0,
            "correct": 1,
            "total": 1,
            "per_task": {},
            "examples": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "reference_set_evaluation": {"enabled": True},
                "experiment": {"output_dir": tmpdir},
            }
            with (
                patch.object(
                    evaluate,
                    "_evaluate_reference_set",
                    return_value=reference_metrics,
                ) as eval_reference,
                patch("builtins.print"),
            ):
                result = evaluate_reference_set(
                    model=FakeEvalModel(),
                    tokenizer=object(),
                    config=config,
                    system_prompt="system",
                    use_system_prompt=False,
                    step=1,
                    checkpoint_task="task_1",
                    seed=33,
                    device="cpu",
                )

            self.assertIs(result, reference_metrics)
            eval_reference.assert_called_once()
            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        tmpdir,
                        "reference_seed_33",
                        "step_1_task_1_reference_metrics.json",
                    )
                )
            )

    def test_evaluate_reference_set_internal_computes_per_task_metrics(self):
        model = FakeEvalModel()
        dataset = FakeDataset(
            [
                {"task_name": "scienceqa", "prompt": "p1", "answer": "A"},
                {"task_id": "task_2", "prompt": "p2", "answer": "3"},
            ]
        )
        config = {
            "reference_set_evaluation": {
                "file": "reference.jsonl",
                "batch_size": 2,
                "max_new_tokens": 2,
            },
            "tasks": {"task_2": {"name": "numgluecm"}},
        }

        with (
            patch.object(evaluate, "load_dataset", return_value=dataset),
            patch.object(evaluate, "tqdm", side_effect=lambda iterable, **_: iterable),
            patch.object(
                evaluate,
                "generate_batch_from_texts",
                return_value=[("A", False), ("4", False)],
            ),
        ):
            metrics = _evaluate_reference_set(
                model=model,
                tokenizer=FakeTokenizer(),
                config=config,
                system_prompt="system",
                use_system_prompt=False,
                step=2,
                checkpoint_task="task_2",
                seed=33,
                device="cpu",
            )

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["per_task"]["scienceqa"]["accuracy"], 1.0)
        self.assertEqual(metrics["per_task"]["numgluecm"]["accuracy"], 0.0)
        self.assertEqual(metrics["examples"][0]["seed"], 33)

    def test_save_reference_results_writes_predictions_metrics_and_scores(self):
        reference_metrics = {
            "seed": 33,
            "step": 2,
            "checkpoint_task": "task_2",
            "accuracy": 0.5,
            "correct": 1,
            "total": 2,
            "per_task": {"scienceqa": {"accuracy": 1.0, "correct": 1, "total": 1}},
            "examples": [{"prompt": "p1"}, {"prompt": "p2"}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_reference_results(reference_metrics, tmpdir, seed=33)

            output_dir = os.path.join(tmpdir, "reference_seed_33")
            with open(
                os.path.join(output_dir, "step_2_task_2_reference_predictions.jsonl"),
                encoding="utf-8",
            ) as f:
                prediction_rows = [json.loads(line) for line in f]

            with open(
                os.path.join(output_dir, "step_2_task_2_reference_metrics.json"),
                encoding="utf-8",
            ) as f:
                metrics = json.load(f)

            with open(
                os.path.join(output_dir, "reference_scores.csv"),
                newline="",
                encoding="utf-8",
            ) as f:
                score_rows = list(csv.DictReader(f))

        self.assertEqual(prediction_rows, [{"prompt": "p1"}, {"prompt": "p2"}])
        self.assertNotIn("examples", metrics)
        self.assertEqual(score_rows[0]["reference_accuracy"], "0.5")


if __name__ == "__main__":
    unittest.main()
