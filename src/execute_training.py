import gc

import yaml
import torch

from tqdm.auto import tqdm
from torch.optim import AdamW

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate_accuracy, save_eval_results, evaluate_checkpoint_on_all_tasks
from utils import get_device, set_seed
from data import ChatSFTDataset, SFTCollator
from constants import DEFAULT_SYSTEM_PROMPT, TASK_PROMPT_MAP


def load_data_loader(task_name: str, data_path: str, tokenizer, shuffle: bool, max_length: int, batch_size: int) -> DataLoader:
    dataset = load_dataset("json", data_files=data_path, split="train")

    sft_dataset = ChatSFTDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        task_prompt=TASK_PROMPT_MAP.get(task_name, ""),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_length=max_length,
        enable_thinking=False,
    )

    return DataLoader(
        sft_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=SFTCollator(tokenizer),
    )


def train_one_task(
    model,
    train_data_loader,
    config,
    device,
):
    num_epochs = config["training"]["num_epochs"]
    learning_rate = float(config["training"].get("learning_rate", 5e-5))
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    max_train_batches = config["training"].get("max_train_batches")  # optional debug

    model.train()
    model.config.use_cache = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_batches = len(train_data_loader)
    if max_train_batches is not None:
        total_batches = min(total_batches, max_train_batches)

    total_steps = num_epochs * total_batches
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        optimizer_step_count = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_data_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            # Scale loss for gradient accumulation.
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1

            epoch_loss += loss.item()
            batch_count += 1
            progress_bar.update(1)

        # Handle leftover gradients if number of batches is not divisible by grad_accum_steps.
        if batch_count % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"loss: {avg_loss:.4f} - "
            f"optimizer steps: {optimizer_step_count}"
        )

    # Drop optimizer state after the task.
    del optimizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

        # evaluate base model
        task_order = config["continual_learning"]["task_order"]

        evaluate_checkpoint_on_all_tasks(
            model=model,
            tokenizer=tokenizer,
            config=config,
            task_order=task_order,
            task_prompt_map=TASK_PROMPT_MAP,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            step=0,
            checkpoint_task="base",
            seed=seed,
            device=device,
            split_file_key=config["experiment"].get("eval_split_file_key", "val_file"),
        )

        num_epochs = config["training"]["num_epochs"]
        for step, task in enumerate(task_order, start=1):
            task_config = config["tasks"][task]
            task_name = task_config['name']

            print(f"Training model on task: {task}-{task_name}")
                        
            train_data_loader = load_data_loader(
                task_name=task_name,
                data_path=task_config["train_file"], 
                tokenizer=tokenizer,
                shuffle=True,
                max_length=config["training"]["max_length"],
                batch_size=config["training"]["batch_size"]
            )
            print(f"Number of training examples for task {task}-{task_name}: {len(train_data_loader.dataset)}")

            # train on current task
            train_one_task(
                model=model,
                train_data_loader=train_data_loader,
                config=config,
                device=device,
            )
            
            # evaluate on all tasks after training on current task
            evaluate_checkpoint_on_all_tasks(
                model=model,
                tokenizer=tokenizer,
                config=config,
                task_order=task_order,
                task_prompt_map=TASK_PROMPT_MAP,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                step=step,
                checkpoint_task=task,
                seed=seed,
                device=device,
                split_file_key=config["experiment"].get("eval_split_file_key", "val_file"),
            )

            # save model checkpoint
            print(f"Saving model checkpoint for task: {task}-{task_name}")
            model_save_path = f"{config['experiment']['output_dir']}/model_{task}_{seed}"
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

            # store config
            with open(f"{model_save_path}/training_config.yaml", "w") as f:
                yaml.dump(config, f)            

            # cleanup resources
            del train_data_loader 

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