import torch
from constants import DEFAULT_SYSTEM_PROMPT, TASK_TRACE_SCIENCEQA, TASK_PROMPT_MAP
from torch.utils.data import Dataset as TorchDataset


def _split_answer(example):
    answer, last = example["answer"].split("\n", 1)
    return {"answer": answer, "reasoning": last}


def get_task_prompt(task_name: str) -> str:
    task_name = task_name.strip().lower()

    if task_name not in TASK_PROMPT_MAP:
        raise KeyError(
            f"No task prompt found for task_name='{task_name}'. "
            f"Available task prompts: {list(TASK_PROMPT_MAP.keys())}"
        )

    return TASK_PROMPT_MAP[task_name]


def build_messages(
    prompt: str,
    task_prompt: str,
    system_prompt: str | None = None,
    use_system_prompt: bool = True,
):
    user_text = f"{task_prompt.strip()}\n\n{prompt.strip()}"

    messages = []

    if use_system_prompt and system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_text})

    return messages



class ChatSFTDataset(TorchDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        task_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        use_system_prompt: bool = False,
        max_length: int = 512,
        enable_thinking: bool = False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.task_prompt = get_task_prompt(self.task_name)
        self.system_prompt = system_prompt
        self.use_system_prompt = use_system_prompt
        self.max_length = max_length
        self.enable_thinking = enable_thinking

        # Special handling for ScienceQA to split answer and reasoning for evaluation purposes.
        if self.task_name == TASK_TRACE_SCIENCEQA:
            self.dataset = self.dataset.map(_split_answer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
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
    
