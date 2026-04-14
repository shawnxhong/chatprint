#!/usr/bin/env python3
"""
LoRA Fine-tuning — backend dispatcher.

Reads `training.backend` from config.yaml and delegates to the matching backend:
  - "cuda"  → training/backends/cuda.py   (Unsloth + bitsandbytes 4-bit QLoRA, NVIDIA only)
  - "intel" → training/backends/intel.py  (HF peft + IPEX, Intel Arc GPU or CPU)
  - "cpu"   → training/backends/intel.py  (same as intel, IPEX disabled)

Usage:
    python training/train_lora.py --config config.yaml [--resume]

Set `training.backend` in config.yaml to switch between backends.
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning — backend dispatcher")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = cfg.get("training", {}).get("backend", "cuda").lower()

    print(f"Training backend: {backend}")

    if backend == "cuda":
        try:
            from training.backends import cuda as backend_module
        except ImportError:
            # Support running as `python training/train_lora.py` from repo root
            sys.path.insert(0, str(Path(__file__).parent))
            from backends import cuda as backend_module
        backend_module.run(cfg, args)

    elif backend in ("intel", "cpu"):
        if backend == "cpu":
            # Force IPEX off so the intel backend goes straight to CPU
            cfg.setdefault("training", {}).setdefault("intel", {})["use_ipex"] = False
        try:
            from training.backends import intel as backend_module
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent))
            from backends import intel as backend_module
        backend_module.run(cfg, args)

    else:
        print(f"ERROR: Unknown backend '{backend}'. Choose from: cuda, intel, cpu")
        print("Set training.backend in config.yaml.")
        sys.exit(1)


if __name__ == "__main__":
    main()
