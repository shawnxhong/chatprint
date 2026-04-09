"""Ollama API client for local model inference."""

import json
from typing import Iterator

import ollama as _ollama


def chat_stream(
    model_name: str,
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
) -> Iterator[str]:
    """
    Stream a chat response from Ollama.

    Args:
        model_name: Ollama model name (e.g. "memorial-bot").
        system_prompt: Full system prompt including RAG memories.
        history: List of (user_message, assistant_response) tuples.
        user_message: The latest user message.

    Yields:
        Text chunks as they stream in.
    """
    messages = [{"role": "system", "content": system_prompt}]

    for user_turn, assistant_turn in history:
        messages.append({"role": "user", "content": user_turn})
        messages.append({"role": "assistant", "content": assistant_turn})

    messages.append({"role": "user", "content": user_message})

    stream = _ollama.chat(model=model_name, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            yield delta


def is_model_available(model_name: str) -> bool:
    """Check if the model is available in Ollama."""
    try:
        models = _ollama.list()
        return any(m["name"].split(":")[0] == model_name for m in models.get("models", []))
    except Exception:
        return False
