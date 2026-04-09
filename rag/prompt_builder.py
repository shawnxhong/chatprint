"""Build system prompts with injected RAG memories."""


def build_system_prompt(
    person_name: str,
    contact_name: str,
    memories: list[dict],
    person_description: str = "",
) -> str:
    """
    Construct a system prompt for the memorial bot.

    Args:
        person_name: The deceased friend's name.
        contact_name: Name of the person currently chatting.
        memories: Retrieved memory chunks from RAG.
        person_description: Optional personality description.
    """
    desc_line = f"\n{person_description}" if person_description else ""

    base = (
        f"你是{person_name}，正在和{contact_name}聊天。{desc_line}\n"
        f"用{person_name}平时的语气、口头禅和说话风格来回复。"
        f"要自然亲切，就像真实聊天一样，不要太正式。"
    )

    if not memories:
        return base

    memory_lines = []
    for m in memories:
        date = m.get("date", "")
        text = m.get("text", "")
        prefix = f"[{date}] " if date else ""
        memory_lines.append(f"- {prefix}{text}")

    memory_block = "\n".join(memory_lines)

    return (
        base + "\n\n"
        "以下是你之前说过或经历过的一些相关内容，供你参考（自然地融入对话，不要生硬引用）：\n"
        + memory_block
    )
