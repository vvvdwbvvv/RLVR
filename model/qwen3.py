import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Qwen3Config, default_data_collator
from trl import SFTTrainer, SFTConfig  # <-- 重點

# 本地模組
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.tokenizer import Tokenizer
from data.dataset import MIMICDataset


# ======================
# Config
# ======================
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SOURCE_CSV = "data/notes_icd_long_sample_100.csv"
TOKENIZER_PATH = "./tokenizer/tokenizer.json"
OUTPUT_DIR = "qwen3-clinical-sft-trl"

BATCH_SIZE = 1
NUM_EPOCHS = 1
MAX_STEPS = 30
LEARNING_RATE = 2e-4
LOGGING_STEPS = 1


def main():
    # ------------------
    # Tokenizer & Dataset（tokenization 已在 Dataset 裡處理）
    # ------------------
    tokenizer = Tokenizer(TOKENIZER_PATH)

    train_dataset = MIMICDataset(
        tokenizer=tokenizer,
        data_path=SOURCE_CSV,
        jsonl_output=None,
        parse_notes=False,
        split="train",
        test_size=20,   # 這裡用「最後 20 筆當 val」的邏輯
    )

    eval_dataset = MIMICDataset(
        tokenizer=tokenizer,
        data_path=SOURCE_CSV,
        jsonl_output=None,
        parse_notes=False,
        split="val",
        test_size=20,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval  samples: {len(eval_dataset)}")

    # ------------------
    # Model
    # ------------------
    torch_dtype = (
        torch.float16
        if (torch.cuda.is_available() or torch.backends.mps.is_available())
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
    )

    config: Qwen3Config = model.config
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # ------------------
    # SFTConfig (取代 TrainingArguments)
    # ------------------
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,

        max_steps=MAX_STEPS,           # >0 時優先
        learning_rate=LEARNING_RATE,

        logging_steps=LOGGING_STEPS,
        logging_dir=str(Path(OUTPUT_DIR) / "logs"),

        eval_steps=LOGGING_STEPS,
        save_steps=MAX_STEPS,
        save_total_limit=1,

        report_to=[],               

        remove_unused_columns=False,   

        fp16=torch.cuda.is_available(),
        bf16=False,
        use_mps_device=torch.backends.mps.is_available(),

        packing=False,                 
        max_length=None,           
    )

    # ------------------
    # SFTTrainer
    # ------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,  # 直接 pad input_ids/labels
        # tokenizer 可以不給，因為我們已經 pre-tokenized
        # 若你想用 HF 的 tokenizer 來決定 pad_token_id 也可以補上
    )

    # ------------------
    # Train
    # ------------------
    print("Starting SFT training with TRL SFTTrainer...")
    train_result = trainer.train()
    print("Training finished.")

    # ------------------
    # Save
    # ------------------
    trainer.save_model(OUTPUT_DIR)

    metrics = train_result.metrics
    metrics["total_trainable_params"] = trainable_params
    metrics_path = Path(OUTPUT_DIR) / "train_metrics.json"
    import json
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
