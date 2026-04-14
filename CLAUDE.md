# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**chatprint** turns a person's WeChat chat history into a persona bot — a memorial chatbot that mimics their speaking style. It fine-tunes a model with LoRA on extracted conversation pairs, augments responses with a RAG memory store (ChromaDB + BGE-M3 embeddings), and serves the result via Gradio + Ollama.

Training supports two backends (set `training.backend` in `config.yaml`):
- `cuda` — Unsloth + bitsandbytes 4-bit QLoRA on Qwen2.5-7B, **NVIDIA GPU required** (12 GB+ VRAM).
- `intel` — **Unsloth + bitsandbytes 4-bit QLoRA on Intel Arc GPU**, confirmed working on Intel Arc (7.96 GB VRAM, bf16, Windows). Uses `pytorch-triton-xpu` for XPU kernels. See Intel setup notes below.

## Pipeline (run in order)

All scripts read from `config.yaml`. Edit that file first — set `person.wxid`, `wechat.data_dir`, and `serving.contacts`.

```bash
# 1. Extract WeChat messages (run on machine where WeChat was installed)
python scripts/01_extract_wechat.py --config config.yaml
# Optional flags: --decrypted-dir <path>  (skip decryption if DBs already decrypted)
#                 --key <hex>             (manual encryption key)

# 2–5. Process data and build RAG corpus
python scripts/02_clean_data.py --config config.yaml
python scripts/03_build_training_data.py --config config.yaml
python scripts/04_build_rag_corpus.py --config config.yaml
python scripts/05_index_memories.py --config config.yaml

# 6. Train — set training.backend in config.yaml first:
#      backend: "cuda"   → NVIDIA GPU (Unsloth + QLoRA, Qwen2.5-7B)
#      backend: "intel"  → Intel Arc GPU (Unsloth + QLoRA via pytorch-triton-xpu)
#      backend: "cpu"    → CPU only (HF peft, no quantization)
#
# Intel Arc users: use the MSVC launcher, not train_lora.py directly.
# See "Intel Arc Setup" section below.
python training/train_lora.py --config config.yaml

# 7. Export to GGUF for Ollama
python scripts/06_export_model.py --config config.yaml
```

## Setup & Running

```bash
# First-time setup (installs Ollama + Python serving deps, registers model)
./install.sh        # Mac/Linux
install.bat         # Windows

# Start the chatbot (http://localhost:7860)
./start.sh          # Mac/Linux
start.bat           # Windows

# Or run directly
python serving/app.py --config config.yaml
```

`install.sh` only installs serving dependencies (`gradio chromadb sentence-transformers ollama pyyaml`). Training dependencies (`unsloth`, `bitsandbytes`, etc.) must be installed separately: `pip install -r requirements.txt`.

## Architecture

```
config.yaml          ← single source of truth for all paths and hyperparameters
scripts/01–06        ← sequential data pipeline (each reads config.yaml)
training/
  train_lora.py      ← entry point; dispatches on config.training.backend
  backends/
    cuda.py          ← Unsloth + bitsandbytes 4-bit QLoRA (NVIDIA)
    intel.py         ← HF peft LoRA (CPU fallback only)
  smoke_test/
    test.py          ← Unsloth QLoRA smoke test (Qwen3-0.6B, LAION dataset, 60 steps)
    run_with_msvc.py ← launcher: sources vcvarsall.bat then runs test.py
rag/
  embedder.py        ← wraps sentence-transformers (BGE-M3)
  retriever.py       ← ChromaDB queries, returns top-k memory chunks
  prompt_builder.py  ← injects retrieved memories into the system prompt (Chinese)
serving/
  ollama_client.py   ← streaming chat via Ollama Python SDK
  app.py             ← Gradio UI; wires contact dropdown → RAG → Ollama → stream
```

**Data flow at inference:**
1. User message → `retrieve()` fetches top-k relevant memory chunks from ChromaDB
2. `build_system_prompt()` constructs a Chinese system prompt with injected memories
3. Streamed response from Ollama (the exported GGUF model)

**Data formats:**
- Raw extraction: one JSON per WeChat conversation (`data/raw/<talker>.json`)
- Training data: ShareGPT format (`{"conversations": [{"from": "human"/"gpt", "value": "..."}]}`)
- RAG corpus: plain text chunks stored in ChromaDB collection `"memories"`

## Intel Arc Setup (Windows)

Unsloth supports Intel Arc GPU via `pytorch-triton-xpu`. Confirmed working on:
- Intel Arc Graphics, 7.96 GB VRAM, Windows 11
- `torch 2.9.0+xpu`, `unsloth 2026.4.4`, `pytorch-triton-xpu 3.5.0`, `bitsandbytes 0.50.0.dev0`

**Install** (conda env `chat`):
```bash
cd unsloth && pip install .[intel-gpu-torch290]
```

**Run** — Triton-XPU needs MSVC to JIT-compile its XPU kernel driver. Always use the wrapper:
```bash
# Sources vcvarsall.bat (VS 2022), then runs training
python smoke_test/run_with_msvc.py          # smoke test
python training/run_with_msvc.py            # TODO: add equivalent for train_lora.py
```

**Why the wrapper is needed:** `pytorch-triton-xpu` calls `shutil.which("cl")` to find MSVC's `cl.EXE`. It only recognises MSVC if the path contains `cl.EXE` (uppercase). VS 2022 `vcvarsall.bat x64` adds the right directory to `PATH`.

**First run is slow:** Triton compiles and caches the XPU driver kernel on the first training step (~56s). Subsequent runs use the cache and run at normal speed (~4s/step for 0.6B model).

**Known issues:**
- `dataset_num_proc` must be `1` — Windows multiprocessing requires `if __name__ == "__main__":` guard (already in `test.py` and `train_lora.py`).
- `use_return_dict` deprecation warning from transformers 5.x — harmless.
- `HF_DATASETS_OFFLINE=1` required if HuggingFace network access is blocked (dataset loads from local cache).

## Key Config Parameters

| Key | Purpose |
|-----|---------|
| `person.wxid` | WeChat ID of the person being modeled |
| `wechat.data_dir` | Path to WeChat data directory |
| `training.backend` | `cuda` (NVIDIA Unsloth QLoRA) / `intel` (Intel Arc Unsloth QLoRA) / `cpu` (HF peft) |
| `model.base_model` | HuggingFace model ID for CUDA/Intel backends (default: Qwen2.5-7B-Instruct) |
| `model.intel_base_model` | Override model for Intel backend — use a smaller model if VRAM is tight (e.g. `unsloth/Qwen3-0.6B-bnb-4bit`) |
| `model.gguf_quantization` | `q4_k_m` (smallest) / `q5_k_m` (balanced) / `q6_k` (best) |
| `rag.top_k` | Memories injected per query (default 5) |
| `serving.contacts` | Dropdown options in the UI (who is chatting) |
| `data.max_gap_seconds` | Seconds of silence that splits a conversation (default 3600) |
