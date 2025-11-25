import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from trl import SFTConfig, SFTTrainer

import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import PROMPT, MIMICDataset

# load configuration with yaml
DEFAULT_CONFIG_PATH = Path(__file__).parent / "mac_config.yaml"

# login to wandb, setup project
wandb.login()
wandb.init(
    project="SmolLM2-Clinical-Instruct", config={"dataset": "MIMICIII", "type": "SFT"}
)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    return config or {}


# parse config
def main():
    parser = argparse.ArgumentParser(
        description="Run a lightweight SFT demo on Mac using the mac_config.yaml spec."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config that defines data/model/training fields.",
    )
    args = parser.parse_args()
    wandb.config.update(args)

    run_config = load_config(args.config)
    save_model = run_config.get("save_model", True)
    data_config = run_config.get("data", {})
    model_config = run_config.get("model", {})
    training_config = run_config.get("training", {})

    model_id = model_config.get("id", "HuggingFaceTB/SmolLM2-360M-Instruct")
    source_csv = data_config.get("source_csv", "data/notes_icd_long_sample_100.csv")
    output_dir = Path(data_config.get("output_dir", "smol2-clinical-instruct"))

    batch_size = training_config.get("batch_size", 1)
    eval_batch_size = training_config.get("eval_batch_size", batch_size)
    num_epochs = training_config.get("num_epochs", 1)
    max_steps = training_config.get("max_steps", 30)
    learning_rate = float(training_config.get("learning_rate", 2e-4))
    logging_steps = training_config.get("logging_steps", 1)
    eval_steps = training_config.get("eval_steps", logging_steps)
    save_steps = training_config.get("save_steps", max_steps)
    save_total_limit = training_config.get("save_total_limit", 1)
    max_grad_norm = training_config.get("max_grad_norm", 1.0)

    print(f"load tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("-" * 40)
    print(PROMPT)
    print("-" * 40)

    # prepare datasets
    train_dataset = MIMICDataset(
        tokenizer=tokenizer,
        data_path=source_csv,
        jsonl_output=None,
        parse_notes=False,
        split="train",
        test_size=20,
    )

    eval_dataset = MIMICDataset(
        tokenizer=tokenizer,
        data_path=source_csv,
        jsonl_output=None,
        parse_notes=False,
        split="val",
        test_size=20,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval  samples: {len(eval_dataset)}")

    torch_dtype = (
        torch.float16
        if (torch.cuda.is_available() or torch.backends.mps.is_available())
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        logging_dir=str(output_dir / "logs"),
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        max_grad_norm=max_grad_norm,
        report_to="wandb",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        bf16=False,
        use_mps_device=torch.backends.mps.is_available(),
        packing=False,
        max_length=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()

    if save_model:
        trainer.save_model(str(output_dir))

    metrics = train_result.metrics
    metrics["total_trainable_params"] = trainable_params
    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
