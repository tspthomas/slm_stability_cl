import torch
from constants import DEFAULT_SYSTEM_PROMPT
from torch.utils.data import Dataset as TorchDataset


class ChatSFTDataset(TorchDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        task_prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_length: int = 512,
        enable_thinking: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_prompt = task_prompt
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.enable_thinking = enable_thinking

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[int(idx)]

        user_text = f"{self.task_prompt}\n\n{ex['prompt'].strip()}"
        answer_text = str(ex["answer"]).strip()

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]

        full_messages = prompt_messages + [
            {"role": "assistant", "content": answer_text},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        ).input_ids

        input_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
        ).input_ids

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            labels = labels[-self.max_length:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class SFTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
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
    
