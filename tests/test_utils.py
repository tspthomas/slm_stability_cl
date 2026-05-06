import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, "./src")
from utils import (
    extract_multiple_choice,
    extract_number,
    get_device,
    is_config_lora,
    normalize_answer,
    normalize_number,
    set_seed,
    strip_generation_artifacts,
)


class TestUtils(unittest.TestCase):

    def test_set_seed(self):
        set_seed(42)
        self.assertEqual(torch.initial_seed(), 42)
        self.assertEqual(np.random.get_state()[1][0], 42)

    def test_get_device(self):
        device = get_device()
        self.assertIn(device, ["cuda", "mps", "cpu"])

    def test_is_config_lora(self):
        config_lora = {"peft": {"method": "lora"}}
        config_not_lora = {"peft": {"method": "other"}}
        config_no_peft = {}
        config_disabled_peft = {"peft": None}
        self.assertTrue(is_config_lora(config_lora))
        self.assertFalse(is_config_lora(config_not_lora))
        self.assertFalse(is_config_lora(config_no_peft))
        self.assertFalse(is_config_lora(config_disabled_peft))

    def test_strip_generation_artifacts(self):
        text = (
            "This is a generated answer. "
            "<think> Some thought process. </think>"
            "**Bold text**<|im_end|><|endoftext|>"
        )
        cleaned = strip_generation_artifacts(text)
        self.assertEqual(cleaned, "This is a generated answer. Bold text")

    def test_strip_generation_artifacts_removes_multiline_thinking(self):
        text = "Answer: A\n<think>\nfirst thought\nsecond thought\n</think>"
        cleaned = strip_generation_artifacts(text)
        self.assertEqual(cleaned, "Answer: A")

    def test_normalize_number(self):
        self.assertEqual(normalize_number("1,234.56"), "1234.56")
        self.assertEqual(normalize_number("  -1234  "), "-1234")
        self.assertEqual(normalize_number("12,34,567"), "1234567")
        self.assertEqual(normalize_number("186.0"), "186")
        self.assertEqual(normalize_number("42."), "42")
        self.assertEqual(normalize_number("June"), "June")

    def test_extract_number(self):
        text = "The answer is 42. <think> Some thought process. </think>"
        extracted = extract_number(text)
        self.assertEqual(extracted, "42")

    def test_extract_number_prefers_final_answer_markers(self):
        text = "We considered 10 and 20. Final answer: 1,234."
        self.assertEqual(extract_number(text), "1234")

    def test_extract_number_falls_back_to_last_number(self):
        text = "First estimate: 5. Revised estimate: 7."
        self.assertEqual(extract_number(text), "7")

    def test_extract_number_returns_clean_text_when_no_number_exists(self):
        self.assertEqual(extract_number("Answer: June"), "ANSWER: JUNE")

    def test_extract_multiple_choice(self):
        text = "The correct option (B). <think> Some thought process. </think>"
        extracted = extract_multiple_choice(text)
        self.assertEqual(extracted, "B")

    def test_extract_multiple_choice_prefers_explicit_marker(self):
        text = "A might look plausible. Final answer: D because of the clue."
        self.assertEqual(extract_multiple_choice(text), "D")

    def test_extract_multiple_choice_uses_last_standalone_option(self):
        text = "The prompt mentions A and B, but the reasoning supports C."
        self.assertEqual(extract_multiple_choice(text), "C")

    def test_normalize_answer(self):
        text_mc = "The correct option is (C). <think> Some thought process. </think>"
        normalized_mc = normalize_answer(text_mc, task_name="scienceqa")
        self.assertEqual(normalized_mc, "C")

        text_num = "The answer is 3.14. <think> Some thought process. </think>"
        normalized_num = normalize_answer(text_num, task_name="numgluecm")
        self.assertEqual(normalized_num, "3.14")

    def test_normalize_answer_uses_generic_final_answer_for_unknown_task(self):
        text = "Reasoning goes here.\nFinal answer: stable"
        self.assertEqual(normalize_answer(text, task_name="unknown"), "STABLE")
