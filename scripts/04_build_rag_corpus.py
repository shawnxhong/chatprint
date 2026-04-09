#!/usr/bin/env python3
"""
Script 04: Build RAG memory corpus from WeChat chat history

Extracts factual/episodic content from the target person's messages
and chunks them for vector search. These memories will be retrieved
at chat time to inject relevant context into the system prompt.

Usage:
    python scripts/04_build_rag_corpus.py --config config.yaml
"""

import argparse
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# Heuristics: messages likely containing memorable/factual content
# (longer messages, ones with location/event/opinion markers)
MEMORY_INDICATORS = [
    r"记得", r"那次", r"那天", r"上次", r"以前", r"之前",  # episodic references
    r"去了", r"去过", r"玩了", r"旅游", r"出差",            # travel/activity
    r"觉得", r"感觉", r"认为", r"觉得", r"看法",            # opinions
    r"喜欢", r"不喜欢", r"讨厌", r"爱",                    # preferences
    r"工作", r"项目", r"公司", r"同事",                      # work
    r"家", r"爸", r"妈", r"兄", r"姐", r"弟", r"妹",        # family
    r"朋友", r"认识",                                       # relationships
    r"生日", r"周年", r"纪念",                               # dates
    r"梦", r"睡", r"起床",                                   # daily life stories
]

MEMORY_RE = re.compile("|".join(MEMORY_INDICATORS))


def is_memorable(content: str, min_length: int = 15) -> bool:
    """Heuristic: is this message worth storing as a memory?"""
    if len(content) < min_length:
        return False
    return bool(MEMORY_RE.search(content))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by sentence boundaries."""
    # Split on Chinese sentence endings
    sentences = re.split(r"([。！？\n])", text)
    # Re-attach the punctuation
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        reconstructed.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ""))
    if len(sentences) % 2 == 1:
        reconstructed.append(sentences[-1])

    chunks = []
    current_chunk = ""
    for sentence in reconstructed:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + sentence
            else:
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) >= 10]


def main():
    parser = argparse.ArgumentParser(description="Build RAG memory corpus")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    person_cfg = cfg["person"]
    data_cfg = cfg["data"]
    rag_cfg = cfg["rag"]

    target_wxid = person_cfg.get("wxid", "")
    chunk_size = rag_cfg.get("chunk_size", 400)
    overlap = rag_cfg.get("chunk_overlap", 50)

    processed_dir = Path(data_cfg["processed_dir"])
    out_dir = Path(data_cfg["rag_corpus_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    contact_names: dict[str, str] = {}
    contacts_file = Path(data_cfg["raw_dir"]) / "contacts.json"
    if contacts_file.exists():
        with open(contacts_file) as f:
            contact_names = json.load(f)

    conv_files = list(processed_dir.glob("*.json"))

    all_chunks: list[dict] = []
    stats = {"messages_scanned": 0, "memorable": 0, "chunks": 0}

    print(f"Building RAG corpus from {len(conv_files)} conversations...")

    for conv_file in tqdm(conv_files):
        with open(conv_file, encoding="utf-8") as f:
            conv = json.load(f)

        talker = conv.get("talker", "")
        contact_name = conv.get("contact_name") or contact_names.get(talker, talker)
        messages = conv.get("messages", [])

        # Collect consecutive target-person messages as "episodes"
        episode_buffer: list[dict] = []

        def flush_episode(buffer: list[dict]):
            if not buffer:
                return
            combined_text = " ".join(m["content"] for m in buffer)
            if not is_memorable(combined_text):
                return

            ts = buffer[0]["timestamp"]
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

            chunks = chunk_text(combined_text, chunk_size, overlap)
            for chunk in chunks:
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        "date": date_str,
                        "timestamp": ts,
                        "contact": contact_name,
                        "talker": talker,
                        "is_group": conv.get("is_group", False),
                    },
                })
            stats["memorable"] += 1
            stats["chunks"] += len(chunks)

        for msg in messages:
            stats["messages_scanned"] += 1
            if msg["sender"] == target_wxid:
                episode_buffer.append(msg)
            else:
                flush_episode(episode_buffer)
                episode_buffer = []

        flush_episode(episode_buffer)

    # Write as JSONL
    out_path = out_dir / "memories.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nResults:")
    print(f"  Messages scanned: {stats['messages_scanned']}")
    print(f"  Memorable episodes: {stats['memorable']}")
    print(f"  Memory chunks written: {stats['chunks']}")
    print(f"  Output: {out_path}")
    print(f"\nNext step: python scripts/05_index_memories.py --config {args.config}")


if __name__ == "__main__":
    main()
