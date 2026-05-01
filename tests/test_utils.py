# create code to test the utils.py file

import unittest

import numpy as np
import torch

import sys
sys.path.insert(0, "./src")
from utils import (
    set_seed,
    get_device,
    is_config_lora,
    strip_generation_artifacts,
    normalize_number,
    extract_number,
    extract_multiple_choice,
    normalize_answer,
    MULTIPLE_CHOICE_TASKS,
    NUMERIC_TASKS,
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
        self.assertTrue(is_config_lora(config_lora))
        self.assertFalse(is_config_lora(config_not_lora))
        self.assertFalse(is_config_lora(config_no_peft))

    def test_strip_generation_artifacts(self):
        text = "This is a generated answer. <think> Some thought process. </think>**Bold text**"
        cleaned = strip_generation_artifacts(text)
        self.assertEqual(cleaned, "This is a generated answer. Bold text")

    def test_normalize_number(self):
        self.assertEqual(normalize_number("1,234.56"), "1234.56")
        self.assertEqual(normalize_number("  -1234  "), "-1234")
        self.assertEqual(normalize_number("12,34,567"), "1234567")

    def test_extract_number(self):
        text = "The answer is 42. <think> Some thought process. </think>"
        extracted = extract_number(text)
        self.assertEqual(extracted, "42")

    def test_extract_multiple_choice(self):
        text = "The correct option (B). <think> Some thought process. </think>"
        extracted = extract_multiple_choice(text)
        self.assertEqual(extracted, "B")

    def test_normalize_answer(self):
        text_mc = "The correct option is (C). <think> Some thought process. </think>"
        normalized_mc = normalize_answer(text_mc, task_name="scienceqa")
        self.assertEqual(normalized_mc, "C")

        text_num = "The answer is 3.14. <think> Some thought process. </think>"
        normalized_num = normalize_answer(text_num, task_name="numgluecm")
        self.assertEqual(normalized_num, "3.14")