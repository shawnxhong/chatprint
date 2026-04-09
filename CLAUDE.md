# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**chatprint** turns a person's WeChat chat history into a persona bot — a memorial chatbot that mimics their speaking style. It fine-tunes a Qwen2.5-7B model with QLoRA on extracted conversation pairs, augments responses with a RAG memory store (ChromaDB + BGE-M3 embeddings), and serves the result via Gradio + Ollama.

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

# 6. Train (requires NVIDIA GPU, 12GB+ VRAM)
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
training/            ← QLoRA fine-tuning via Unsloth on Qwen2.5-7B-Instruct
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

## Key Config Parameters

| Key | Purpose |
|-----|---------|
| `person.wxid` | WeChat ID of the person being modeled |
| `wechat.data_dir` | Path to WeChat data directory |
| `model.gguf_quantization` | `q4_k_m` (smallest) / `q5_k_m` (balanced) / `q6_k` (best) |
| `rag.top_k` | Memories injected per query (default 5) |
| `serving.contacts` | Dropdown options in the UI (who is chatting) |
| `data.max_gap_seconds` | Seconds of silence that splits a conversation (default 3600) |
