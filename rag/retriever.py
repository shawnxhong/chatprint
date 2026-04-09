"""ChromaDB-backed memory retriever."""

from functools import lru_cache
from typing import Optional

import chromadb

from rag.embedder import embed


@lru_cache(maxsize=1)
def get_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


def retrieve(
    query: str,
    chroma_dir: str,
    collection_name: str,
    top_k: int = 5,
    contact_filter: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve top-k relevant memory chunks for a query.

    Args:
        query: The user's message to search for relevant memories.
        chroma_dir: Path to ChromaDB storage directory.
        collection_name: Name of the ChromaDB collection.
        top_k: Number of results to return.
        contact_filter: If set, only return memories from conversations with this contact.

    Returns:
        List of dicts with 'text', 'date', 'contact' keys.
    """
    collection = get_collection(chroma_dir, collection_name)

    if collection.count() == 0:
        return []

    query_embedding = embed([query])[0].tolist()

    where_filter = None
    if contact_filter:
        where_filter = {"contact": {"$eq": contact_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    memories = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Distance is cosine distance (0 = identical, 2 = opposite)
        # Filter out low-relevance results
        if dist > 1.2:
            continue
        memories.append({
            "text": doc,
            "date": meta.get("date", ""),
            "contact": meta.get("contact", ""),
            "relevance": round(1 - dist / 2, 3),  # Convert to 0-1 similarity
        })

    return memories
