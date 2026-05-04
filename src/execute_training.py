import gc

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import DEFAULT_SYSTEM_PROMPT
from data import ChatSFTDataset, SFTCollator
from evaluate import evaluate_checkpoint_on_all_tasks, evaluate_reference_set
from stability import compute_reference_stability_metrics
from train import build_lora_model, train_one_task
from utils import get_device, is_config_lora, set_seed


def load_data_loader(
    task_name: str,
    data_path: str,
    tokenizer,
    shuffle: bool,
    max_length: int,
    batch_size: int,
) -> DataLoader:
    dataset = load_dataset("json", data_files=data_path, split="train")

    sft_dataset = ChatSFTDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        task_name=task_name,
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

    use_system_prompt = config["experiment"].get("system_prompt", False)
    print(f"Using system prompt: {use_system_prompt}")

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
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            use_system_prompt=use_system_prompt,
            step=0,
            checkpoint_task="base",
            seed=seed,
            device=device,
        )

        evaluate_reference_set(
            model=model,
            tokenizer=tokenizer,
            config=config,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            use_system_prompt=use_system_prompt,
            step=0,
            checkpoint_task="base",
            seed=seed,
            device=device,
        )

        compute_reference_stability_metrics(
            model=model,
            tokenizer=tokenizer,
            config=config,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            use_system_prompt=use_system_prompt,
            device=device,
            seed=seed,
            step=0,
            checkpoint_task="base",
        )

        # optionally wrap with LoRA for parameter-efficient fine-tuning
        if is_config_lora(config):
            print("Wrapping model with LoRA for parameter-efficient fine-tuning.")
            model = build_lora_model(model, config)

        # print number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable parameters: {trainable_params} / {total_params} ({trainable_params / total_params:.2%})"
        )

        num_epochs = config["training"]["num_epochs"]
        for step, task in enumerate(task_order, start=1):
            task_config = config["tasks"][task]
            task_name = task_config["name"]

            print(f"Training model on task: {task}-{task_name}")

            train_data_loader = load_data_loader(
                task_name=task_name,
                data_path=task_config["train_file"],
                tokenizer=tokenizer,
                shuffle=True,
                max_length=config["training"]["max_length"],
                batch_size=config["training"]["batch_size"],
            )
            print(
                f"Number of training examples for task {task}-{task_name}: {len(train_data_loader.dataset)}"
            )

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
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                use_system_prompt=use_system_prompt,
                step=step,
                checkpoint_task=task,
                seed=seed,
                device=device,
            )

            # evaluate reference set after training on current task
            evaluate_reference_set(
                model=model,
                tokenizer=tokenizer,
                config=config,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                use_system_prompt=use_system_prompt,
                step=step,
                checkpoint_task=task,
                seed=seed,
                device=device,
            )

            # compute stability metrics on reference set after training on current task
            compute_reference_stability_metrics(
                model=model,
                tokenizer=tokenizer,
                config=config,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                use_system_prompt=use_system_prompt,
                device=device,
                seed=seed,
                step=step,
                checkpoint_task=task,
            )

            # save model checkpoint
            print(f"Saving model checkpoint for task: {task}-{task_name}")
            model_save_path = (
                f"{config['experiment']['output_dir']}/model_{task}_{seed}"
            )
            model_save_path = (
                model_save_path + "_lora" if is_config_lora(config) else model_save_path
            )

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

    parser = argparse.ArgumentParser(
        description="Run longitudinal personalization experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    main(args.config)
