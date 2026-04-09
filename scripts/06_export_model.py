#!/usr/bin/env python3
"""
Script 06: Export fine-tuned model to GGUF for Ollama deployment

Merges the LoRA adapter into the base model and exports to GGUF format.
The resulting .gguf file can be used with Ollama on any platform.

Usage:
    python scripts/06_export_model.py --config config.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    person_cfg = cfg["person"]
    serving_cfg = cfg["serving"]

    adapter_dir = Path(model_cfg["lora_adapter_dir"])
    gguf_dir = Path(model_cfg["gguf_dir"])
    quantization = model_cfg.get("gguf_quantization", "q4_k_m")
    person_name = person_cfg.get("name", "friend")
    ollama_name = serving_cfg.get("ollama_model_name", "memorial-bot")

    gguf_dir.mkdir(parents=True, exist_ok=True)

    if not (adapter_dir / "adapter_config.json").exists():
        print(f"ERROR: No LoRA adapter found at {adapter_dir}")
        print("Run training/train_lora.py first.")
        sys.exit(1)

    # Use Unsloth to merge and export
    print("Loading fine-tuned model with Unsloth...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run: pip install unsloth")
        sys.exit(1)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        load_in_4bit=False,  # Load in full precision for export
    )

    gguf_path = gguf_dir / f"{ollama_name}.gguf"
    print(f"Exporting to GGUF ({quantization})...")
    model.save_pretrained_gguf(
        str(gguf_dir / ollama_name),
        tokenizer,
        quantization_method=quantization,
    )

    # Find the actual GGUF file (unsloth may add suffix)
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        print("ERROR: GGUF export failed — no .gguf file found.")
        sys.exit(1)

    gguf_path = gguf_files[0]
    print(f"GGUF saved to: {gguf_path}")

    # Write Ollama Modelfile
    modelfile_path = Path("serving") / "Modelfile"
    system_prompt = (
        f"你是{person_name}。用{person_name}平时的语气和风格来聊天，"
        f"要自然、亲切，就像真实聊天一样。"
    )
    modelfile_content = f"""FROM {gguf_path.resolve()}

PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM \"\"\"{system_prompt}\"\"\"
"""
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    print(f"Ollama Modelfile written to: {modelfile_path}")

    # Register with Ollama if available
    print(f"\nRegistering model with Ollama as '{ollama_name}'...")
    result = subprocess.run(
        ["ollama", "create", ollama_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Model registered. Test with: ollama run {ollama_name}")
    else:
        print("WARNING: Ollama registration failed (is Ollama installed and running?)")
        print(f"  To register manually: ollama create {ollama_name} -f {modelfile_path}")
        print(f"  Error: {result.stderr.strip()}")

    print(f"\nNext step: python serving/app.py --config config.yaml")


if __name__ == "__main__":
    main()
