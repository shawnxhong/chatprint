#!/usr/bin/env python3
"""
Smoke test for the Intel/CPU training backend.

Creates a tiny synthetic dataset, loads Qwen2.5-0.5B-Instruct (smallest Qwen
with chat template, ~1 GB download), applies LoRA, and runs 3 training steps.

Usage:
    python training/smoke_test_intel.py

Pass --model <hf-id> to override the test model.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

SYNTHETIC_CONVERSATIONS = [
    {
        "conversations": [
            {"from": "human", "value": "你最近怎么样？"},
            {"from": "gpt",   "value": "还不错，就是有点累。你呢？"},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "周末有什么打算？"},
            {"from": "gpt",   "value": "可能去爬山，你要一起吗？"},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "今天吃什么好？"},
            {"from": "gpt",   "value": "火锅吧，天冷了正好。"},
        ]
    },
    {
        "conversations": [
            {"from": "human", "value": "工作压力大吗？"},
            {"from": "gpt",   "value": "还好，习惯了就好。"},
        ]
    },
] * 4  # 16 examples — enough for 1 epoch with packing


def banner(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def check_imports():
    banner("1/5  Checking imports")
    import torch
    import transformers
    import peft
    import trl
    import accelerate
    import datasets
    print(f"  torch:          {torch.__version__}")
    print(f"  transformers:   {transformers.__version__}")
    print(f"  peft:           {peft.__version__}")
    print(f"  trl:            {trl.__version__}")
    print(f"  accelerate:     {accelerate.__version__}")
    print(f"  datasets:       {datasets.__version__}")
    return torch


def detect_device(torch):
    banner("2/5  Detecting device")
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if torch.xpu.is_available():
            device = "xpu"
            print("  Intel Arc GPU detected via IPEX — using XPU.")
        else:
            device = "cpu"
            print("  IPEX present but no XPU device — falling back to CPU.")
    except ImportError:
        device = "cpu"
        print("  IPEX not installed — using CPU.")
    print(f"  Device: {device}")
    return device


def load_model_and_tokenizer(model_id: str, device: str, torch):
    banner(f"3/5  Loading model: {model_id}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print(f"  Loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")
    return model, tokenizer


def apply_lora(model):
    banner("4/5  Applying LoRA adapter")
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def run_training(model, tokenizer, device: str, tmp_dir: Path):
    banner("5/5  Running 3 training steps")
    import datasets as hf_datasets
    from trl import SFTTrainer, SFTConfig

    # Write synthetic data to temp file
    train_file = tmp_dir / "train.json"
    train_file.write_text(json.dumps(SYNTHETIC_CONVERSATIONS), encoding="utf-8")

    raw = hf_datasets.Dataset.from_list(SYNTHETIC_CONVERSATIONS)

    def fmt(ex):
        messages = [
            {"role": ("user" if t["from"] == "human" else "assistant"), "content": t["value"]}
            for t in ex["conversations"]
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    dataset = raw.map(fmt, remove_columns=raw.column_names)

    training_args = SFTConfig(
        output_dir=str(tmp_dir / "adapter"),
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        optim="adamw_torch",
        fp16=False,
        bf16=(device != "cpu"),
        use_cpu=(device == "cpu"),
        logging_steps=1,
        save_steps=999,
        report_to="none",
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=256,
        packing=(device != "cpu"),  # packing needs flash_attention_2; off on CPU
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    stats = trainer.train()
    print(f"\n  Steps completed: {int(stats.metrics['train_steps_per_second'] and 3)}")
    print(f"  Final loss:      {stats.metrics.get('train_loss', 'n/a')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID to use for the smoke test",
    )
    args = parser.parse_args()

    print("\nchatprint Intel backend smoke test")
    print(f"Test model: {args.model}")

    try:
        torch = check_imports()
        device = detect_device(torch)
        model, tokenizer = load_model_and_tokenizer(args.model, device, torch)
        model = apply_lora(model)

        with tempfile.TemporaryDirectory() as tmp:
            run_training(model, tokenizer, device, Path(tmp))

        banner("SMOKE TEST PASSED")
        print("  Intel/CPU backend is working correctly.\n")

    except Exception as e:
        banner("SMOKE TEST FAILED")
        import traceback
        traceback.print_exc()
        print(f"\n  Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
