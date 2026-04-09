#!/usr/bin/env python3
"""
Script 05: Embed and index memory chunks into ChromaDB

Reads the JSONL corpus from 04_build_rag_corpus.py, embeds each chunk
with BAAI/bge-m3, and stores in a local ChromaDB collection.

Usage:
    python scripts/05_index_memories.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Index memories into ChromaDB")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the collection")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    rag_cfg = cfg["rag"]

    corpus_path = Path(data_cfg["rag_corpus_dir"]) / "memories.jsonl"
    chroma_dir = data_cfg["chroma_db_dir"]
    collection_name = rag_cfg["collection_name"]
    embedding_model_name = rag_cfg["embedding_model"]

    if not corpus_path.exists():
        print(f"ERROR: {corpus_path} not found. Run 04_build_rag_corpus.py first.")
        return

    # Load chunks
    chunks = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Loaded {len(chunks)} memory chunks.")

    # Initialize embedding model
    print(f"Loading embedding model: {embedding_model_name}")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(embedding_model_name)

    # Initialize ChromaDB
    import chromadb
    client = chromadb.PersistentClient(path=chroma_dir)

    if args.reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Check what's already indexed
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    print(f"Already indexed: {len(existing_ids)} | New to index: {len(new_chunks)}")

    if not new_chunks:
        print("Nothing to index. Use --reset to re-index everything.")
        return

    # Embed and index in batches
    print("Embedding and indexing...")
    for i in tqdm(range(0, len(new_chunks), args.batch_size)):
        batch = new_chunks[i:i + args.batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    total = collection.count()
    print(f"\nDone. Total indexed: {total} memories in '{collection_name}'")
    print(f"ChromaDB stored at: {chroma_dir}")
    print(f"\nNext step: python training/train_lora.py --config config.yaml")


if __name__ == "__main__":
    main()
