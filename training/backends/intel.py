"""
Intel GPU / CPU backend for LoRA fine-tuning.

Uses standard HuggingFace transformers + peft + trl (no Unsloth, no bitsandbytes).
Supports Intel Arc GPU via intel_extension_for_pytorch (IPEX); falls back to CPU
if IPEX is not installed.

Requirements:
    pip install transformers peft trl accelerate datasets torch

Optional (Intel Arc GPU acceleration):
    Install intel_extension_for_pytorch matching your torch version:
    https://intel.github.io/intel-extension-for-pytorch/
"""

import json
from pathlib import Path


def load_dataset_from_json(path: str):
    """Load ShareGPT format JSON as HuggingFace Dataset."""
    from datasets import Dataset

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def format_conversation(example, tokenizer):
    """Apply Qwen ChatML template to a ShareGPT conversation."""
    conversations = example["conversations"]
    messages = []
    for turn in conversations:
        role_map = {"system": "system", "human": "user", "gpt": "assistant"}
        role = role_map.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def _resolve_device(use_ipex: bool) -> str:
    """Return the best available device string for Intel training."""
    import torch

    if use_ipex:
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if torch.xpu.is_available():
                print("IPEX detected — using Intel XPU (Arc GPU).")
                return "xpu"
            else:
                print("IPEX installed but no XPU device found — falling back to CPU.")
        except ImportError:
            print("intel_extension_for_pytorch not found — falling back to CPU.")
            print("To enable Intel Arc GPU: install IPEX matching your torch version.")
            print("See: https://intel.github.io/intel-extension-for-pytorch/")
    else:
        print("use_ipex=false in config — using CPU.")

    return "cpu"


def _merge_intel_overrides(train_cfg: dict) -> dict:
    """Return a copy of train_cfg with intel-specific overrides applied."""
    merged = dict(train_cfg)
    intel_overrides = train_cfg.get("intel", {})
    merged.update(intel_overrides)
    return merged


def run(cfg: dict, args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    model_cfg = cfg["model"]
    train_cfg = _merge_intel_overrides(cfg["training"])
    data_cfg = cfg["data"]

    intel_overrides = cfg["training"].get("intel", {})
    use_ipex = intel_overrides.get("use_ipex", True)
    use_bf16 = intel_overrides.get("bf16", True)

    base_model = model_cfg.get("intel_base_model", "Qwen/Qwen3.5-2B-Instruct")
    adapter_dir = model_cfg["lora_adapter_dir"]
    train_path = Path(data_cfg["training_dir"]) / "train.json"
    val_path = Path(data_cfg["training_dir"]) / "val.json"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run scripts/03_build_training_data.py first.")
        return

    device = _resolve_device(use_ipex)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Loading base model: {base_model}")
    print(f"  Device:    {device}")
    print(f"  Precision: {'bf16' if use_bf16 else 'fp32'}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=model_cfg["base_model_dir"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch_dtype,
        cache_dir=model_cfg["base_model_dir"],
        trust_remote_code=True,
    )
    model = model.to(device)

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        print(f"Val examples:   {len(val_dataset)}")

    from trl import SFTTrainer, SFTConfig

    # trl 1.x: SFT-specific settings (dataset_text_field, max_length, packing)
    # live in SFTConfig, not TrainingArguments. adamw_8bit is bitsandbytes-only.
    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        fp16=False,
        bf16=(use_bf16 and device != "cpu"),  # bf16 on XPU only; CPU uses fp32
        use_cpu=(device == "cpu"),
        logging_steps=10,
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"] if val_dataset else None,
        eval_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=0,  # 0 avoids multiprocessing issues on Windows/CPU
        # SFT-specific
        dataset_text_field="text",
        max_length=train_cfg["max_seq_length"],
        dataset_num_proc=1,
        # Packing requires flash_attention_2; disable on CPU to avoid
        # cross-sample contamination with standard attention kernels.
        packing=(device != "cpu"),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # trl 1.x replaces tokenizer= with processing_class=
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    print("\nStarting training...")
    print(f"  Model:           {base_model}")
    print(f"  Backend:         Intel/CPU (HF peft, no quantization)")
    print(f"  Device:          {device}")
    print(f"  Epochs:          {train_cfg['epochs']}")
    print(f"  Effective batch: {train_cfg['per_device_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Max seq length:  {train_cfg['max_seq_length']}")
    print(f"  Output:          {adapter_dir}\n")

    trainer_stats = trainer.train(resume_from_checkpoint=args.resume)

    print(f"\nTraining complete!")
    print(f"  Runtime:   {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"  Samples/s: {trainer_stats.metrics['train_samples_per_second']:.2f}")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")
    print(f"\nNext step: python scripts/06_export_model.py --config config.yaml")
