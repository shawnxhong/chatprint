#!/usr/bin/env python3
"""
QLoRA Fine-tuning with Unsloth

Trains a LoRA adapter on top of Qwen2.5-7B-Instruct using the ShareGPT
conversation pairs built by 03_build_training_data.py.

Usage:
    python training/train_lora.py --config config.yaml

Requirements:
    - NVIDIA GPU with 12GB+ VRAM (RTX 4070 or better)
    - pip install unsloth bitsandbytes transformers datasets trl peft accelerate
"""

import argparse
import json
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_json(path: str):
    """Load ShareGPT format JSON as HuggingFace Dataset."""
    from datasets import Dataset

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_conversation(example, tokenizer):
    """Apply Qwen ChatML template to a ShareGPT conversation."""
    conversations = example["conversations"]

    # Convert ShareGPT format to HuggingFace messages format
    messages = []
    for turn in conversations:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})

    # Apply the model's chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning with Unsloth")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    base_model = model_cfg["base_model"]
    adapter_dir = model_cfg["lora_adapter_dir"]
    train_path = Path(data_cfg["training_dir"]) / "train.json"
    val_path = Path(data_cfg["training_dir"]) / "val.json"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run scripts/03_build_training_data.py first.")
        return

    print(f"Loading base model: {base_model}")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed.")
        print("Install: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=train_cfg["max_seq_length"],
        dtype=None,         # Auto-detect
        load_in_4bit=True,  # QLoRA
        cache_dir=model_cfg["base_model_dir"],
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"LoRA config: r={train_cfg['lora_r']}, alpha={train_cfg['lora_alpha']}")

    # Load and format datasets
    print("Loading training data...")
    train_dataset = load_dataset_from_json(str(train_path))
    val_dataset = load_dataset_from_json(str(val_path)) if val_path.exists() else None

    train_dataset = train_dataset.map(
        lambda ex: format_conversation(ex, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    if val_dataset:
        val_dataset = val_dataset.map(
            lambda ex: format_conversation(ex, tokenizer),
            remove_columns=val_dataset.column_names,
        )

    print(f"Train examples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val examples: {len(val_dataset)}")

    # Training
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=adapter_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=False,
        bf16=True,   # Use bf16 on Ampere+ GPUs (RTX 3xxx/4xxx)
        logging_steps=10,
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"] if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
        resume_from_checkpoint=args.resume,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_length"],
        dataset_num_proc=2,
        packing=True,   # Pack short sequences for efficiency
        args=training_args,
    )

    print("\nStarting training...")
    print(f"  Model: {base_model}")
    print(f"  Epochs: {train_cfg['epochs']}")
    print(f"  Effective batch size: {train_cfg['per_device_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Output: {adapter_dir}\n")

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume)

    print(f"\nTraining complete!")
    print(f"  Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"  Samples/s: {trainer_stats.metrics['train_samples_per_second']:.2f}")

    # Save the final adapter
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    print(f"\nNext step: python scripts/06_export_model.py --config config.yaml")


if __name__ == "__main__":
    main()
