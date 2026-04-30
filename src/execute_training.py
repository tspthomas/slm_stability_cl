import gc

import yaml
import torch

from tqdm.auto import tqdm
from torch.optim import AdamW

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate_accuracy, save_eval_results
from utils import get_device, set_seed
from data import ChatSFTDataset, SFTCollator
from constants import DEFAULT_SYSTEM_PROMPT, QWEN_MATH_PROMPT, QWEN_MULTIPLE_CHOICE_PROMPT


TASK_PROMPT_MAP = {
    "scienceqa": QWEN_MULTIPLE_CHOICE_PROMPT,
    "fomc": QWEN_MULTIPLE_CHOICE_PROMPT,
    "numglue": QWEN_MATH_PROMPT,
}


def load_data_loader(task_name: str, data_path: str, tokenizer, shuffle: bool) -> DataLoader:
    dataset = load_dataset("json", data_files=data_path, split="train")

    sft_dataset = ChatSFTDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        task_prompt=TASK_PROMPT_MAP.get(task_name, ""),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_length=512,
        enable_thinking=False,
    )

    return DataLoader(
        sft_dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=SFTCollator(tokenizer),
    )


def main(config_path: str):
    """Main function to run the longitudinal personalization experiment based on the provided configuration.

    Args:
        config_path (str): Path to the YAML configuration file containing experiment settings.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Allow seed to be int or list
    config_seeds = config["experiment"]["seed"]
    seeds = config_seeds if isinstance(config_seeds, list) else [config_seeds]

    device = get_device()
    print(f"Device: {device}")

    for seed in seeds:
        print(f"Running experiment with seed: {seed}")
        set_seed(seed)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # load pretrained model
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["name"],
            trust_remote_code=config["model"]["trust_remote_code"],
            torch_dtype=config["model"]["torch_dtype"],
        )
        model.to(device)

        # build trainable model

        # build optimizer
        optimizer = AdamW(model.parameters(), lr=5e-5)

        num_epochs = config["experiment"]["num_epochs"]
        for task in config["continual_learning"]["task_order"]:
            task_config = config["tasks"][task]
            task_name = task_config['name']
            
            # train model
            print(f"Training model on task: {task}-{task_name}")
            train_data_loader = load_data_loader(
                task_name=task_name,
                data_path=task_config["train_file"], 
                tokenizer=tokenizer,
                shuffle=True
            )
            print(f"Number of training examples for task {task}-{task_name}: {len(train_data_loader.dataset)}")

            num_training_steps = num_epochs * len(train_data_loader)
            progress_bar = tqdm(range(num_training_steps))

            model.train()
            model.config.use_cache = False
            for epoch in range(num_epochs):
                epoch_loss = 0.0

                count = 0
                for batch in train_data_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}

                    outputs = model(**batch)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    epoch_loss += loss.item()
                    progress_bar.update(1)

                    count += 1
                    if count == 5:
                        break

                avg_loss = epoch_loss / len(train_data_loader)
                print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}")

            # eval mode on validation set of current task
            print(f"Evaluating model on validation set of task: {task}-{task_name}")
            val_metrics = evaluate_accuracy(
                model=model,
                tokenizer=tokenizer,
                data_file=task_config["val_file"],
                task_name=task_name,
                task_prompt=TASK_PROMPT_MAP.get(task_name, ""),
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                device=device,
                max_examples=config["experiment"].get("max_eval_examples"),
                max_new_tokens=16,
            )

            print(
                f"Validation accuracy for {task}-{task_name}: "
                f"{val_metrics['accuracy']:.4f} "
                f"({val_metrics['correct']}/{val_metrics['total']})"
            )

            # save eval results
            save_eval_results(
                val_metrics=val_metrics,
                output_dir=config["experiment"]["output_dir"],
                task=task,
                seed=seed,
                split_name="val",
            )

            # save model checkpoint
            print(f"Saving model checkpoint for task: {task}-{task_name}")
            model_save_path = f"{config['experiment']['output_dir']}/model_{task}_{seed}"
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

            # cleanup resources
            del train_data_loader
            del val_metrics

            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run longitudinal personalization experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)