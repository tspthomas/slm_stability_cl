import yaml
import torch
from utils import get_device, set_seed
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


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

        # load pretrained model
        model = Qwen3_5ForCausalLM.from_pretrained(
            config["model"]["name"],
            trust_remote_code=config["model"]["trust_remote_code"],
            torch_dtype=config["model"]["torch_dtype"],
        )
        model.to(device)

        # build trainable model

        # build optimizer

        for task in config["continual_learning"]["task_order"]:
            # train model
            print(f"Training model on task: {task}")

            # eval mode on validation set of current task
            print(f"Evaluating model on validation set of task: {task}")

            # save model checkpoint
            print(f"Saving model checkpoint for task: {task}")

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