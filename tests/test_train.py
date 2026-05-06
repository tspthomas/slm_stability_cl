import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import train
from constants import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES
from train import build_lora_model, train_one_task


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=True)
        self.forward_calls = 0
        self.gradient_checkpointing_enabled = False
        self.input_require_grads_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True

    def print_trainable_parameters(self):
        self.print_trainable_parameters_called = True

    def forward(self, input_ids, labels=None, attention_mask=None):
        self.forward_calls += 1
        loss = self.weight * input_ids.float().sum()
        return SimpleNamespace(loss=loss)


class FakePeftModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.print_trainable_parameters_called = False

    def print_trainable_parameters(self):
        self.print_trainable_parameters_called = True


class RecordingAdamW:
    instances = []

    def __init__(self, params, lr, weight_decay):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_calls = 0
        self.zero_grad_calls = []
        RecordingAdamW.instances.append(self)

    def zero_grad(self, set_to_none=True):
        self.zero_grad_calls.append(set_to_none)

    def step(self):
        self.step_calls += 1


class FakeProgressBar:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable
        self.kwargs = kwargs
        self.update_calls = 0

    def update(self, value):
        self.update_calls += value


class TestTrain(unittest.TestCase):
    def setUp(self):
        RecordingAdamW.instances = []

    def _batch(self, value):
        return {
            "input_ids": torch.tensor([[value]], dtype=torch.long),
            "labels": torch.tensor([[value]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def test_build_lora_model_uses_project_defaults(self):
        model = TinyTrainModel()
        captured = {}

        def fake_lora_config(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        def fake_get_peft_model(base_model, peft_config):
            captured["base_model"] = base_model
            captured["peft_config"] = peft_config
            return FakePeftModel(base_model)

        with (
            patch.object(train, "LoraConfig", side_effect=fake_lora_config),
            patch.object(train, "get_peft_model", side_effect=fake_get_peft_model),
        ):
            wrapped_model = build_lora_model(model, {"peft": {"method": "lora"}})

        self.assertIs(captured["base_model"], model)
        self.assertEqual(captured["r"], LORA_RANK)
        self.assertEqual(captured["lora_alpha"], LORA_ALPHA)
        self.assertEqual(captured["lora_dropout"], LORA_DROPOUT)
        self.assertEqual(captured["bias"], "none")
        self.assertEqual(captured["task_type"], train.TaskType.CAUSAL_LM)
        self.assertEqual(captured["target_modules"], LORA_TARGET_MODULES)
        self.assertTrue(wrapped_model.print_trainable_parameters_called)

    def test_build_lora_model_uses_config_overrides(self):
        captured = {}

        def fake_lora_config(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        with (
            patch.object(train, "LoraConfig", side_effect=fake_lora_config),
            patch.object(train, "get_peft_model", side_effect=lambda model, _: model),
        ):
            model = build_lora_model(
                TinyTrainModel(),
                {
                    "peft": {
                        "method": "lora",
                        "r": 8,
                        "lora_alpha": 32,
                        "lora_dropout": 0.2,
                        "bias": "all",
                        "target_modules": ["q_proj"],
                    }
                },
            )

        self.assertTrue(model.print_trainable_parameters_called)
        self.assertEqual(captured["r"], 8)
        self.assertEqual(captured["lora_alpha"], 32)
        self.assertEqual(captured["lora_dropout"], 0.2)
        self.assertEqual(captured["bias"], "all")
        self.assertEqual(captured["target_modules"], ["q_proj"])

    def test_train_one_task_limits_batches_and_disables_cache(self):
        model = TinyTrainModel()
        data_loader = [self._batch(1), self._batch(2), self._batch(3)]
        config = {
            "training": {
                "num_epochs": 1,
                "max_train_batches": 2,
                "gradient_accumulation_steps": 1,
            }
        }

        with (
            patch.object(train, "AdamW", RecordingAdamW),
            patch.object(train, "tqdm", side_effect=FakeProgressBar),
            patch("builtins.print"),
        ):
            train_one_task(model, data_loader, config, device="cpu")

        self.assertFalse(model.config.use_cache)
        self.assertEqual(model.forward_calls, 2)
        self.assertEqual(RecordingAdamW.instances[0].step_calls, 2)

    def test_train_one_task_steps_leftover_accumulated_gradients(self):
        model = TinyTrainModel()
        data_loader = [self._batch(1), self._batch(2), self._batch(3)]
        config = {
            "training": {
                "num_epochs": 1,
                "gradient_accumulation_steps": 2,
            }
        }

        with (
            patch.object(train, "AdamW", RecordingAdamW),
            patch.object(train, "tqdm", side_effect=FakeProgressBar),
            patch("builtins.print"),
        ):
            train_one_task(model, data_loader, config, device="cpu")

        self.assertEqual(model.forward_calls, 3)
        self.assertEqual(RecordingAdamW.instances[0].step_calls, 2)

    def test_train_one_task_enables_checkpointing_and_lora_input_grads(self):
        model = TinyTrainModel()
        config = {
            "peft": {"method": "lora"},
            "training": {
                "num_epochs": 1,
                "gradient_checkpointing": True,
            },
        }

        with (
            patch.object(train, "AdamW", RecordingAdamW),
            patch.object(train, "tqdm", side_effect=FakeProgressBar),
            patch("builtins.print"),
        ):
            train_one_task(model, [self._batch(1)], config, device="cpu")

        self.assertTrue(model.gradient_checkpointing_enabled)
        self.assertTrue(model.input_require_grads_enabled)


if __name__ == "__main__":
    unittest.main()
