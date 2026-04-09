#!/usr/bin/env python3
"""
Script 02: Clean and normalize WeChat messages

Reads raw JSON exports from 01_extract_wechat.py, filters to usable text messages,
merges consecutive messages from the same sender, and outputs cleaned conversation files.

Usage:
    python scripts/02_clean_data.py --config config.yaml
"""

import argparse
import json
import re
from pathlib import Path

import yaml
from tqdm import tqdm


# WeChat message types
MSG_TYPE_TEXT = 1
MSG_TYPE_IMAGE = 3
MSG_TYPE_VOICE = 34
MSG_TYPE_VIDEO = 43
MSG_TYPE_EMOJI = 47
MSG_TYPE_LINK = 49
MSG_TYPE_SYSTEM = 10000

# System message patterns to skip
SYSTEM_PATTERNS = [
    r"^「.*」$",                  # Quoted messages (WeChat recall indicator)
    r"撤回了一条消息",             # Message recall
    r"邀请.*加入了群聊",           # Group invite
    r"修改了群名称",               # Group rename
    r"移出了群聊",                 # Removed from group
    r"以下为新消息",               # New message divider
    r"^<msg>",                    # Raw XML (not parsed text)
    r"^\[.*\]$",                  # [图片], [语音], [视频], [表情], etc.
]

SYSTEM_RE = re.compile("|".join(SYSTEM_PATTERNS))


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_usable_message(msg: dict) -> bool:
    """Return True if this message should be included in training data."""
    msg_type = msg.get("type", 0)
    content = (msg.get("content") or "").strip()

    # Only text messages
    if msg_type != MSG_TYPE_TEXT:
        return False

    # Skip empty content
    if not content:
        return False

    # Skip system/automated messages
    if SYSTEM_RE.search(content):
        return False

    # Skip very short noise (single punctuation, etc.)
    if len(content) < 1:
        return False

    return True


def merge_consecutive_turns(messages: list[dict], max_gap_seconds: int) -> list[dict]:
    """
    Merge consecutive messages from the same sender into one turn.
    Breaks on sender change or time gap exceeding max_gap_seconds.
    """
    if not messages:
        return []

    merged = []
    current = messages[0].copy()

    for msg in messages[1:]:
        same_sender = msg["sender"] == current["sender"]
        time_gap = msg["timestamp"] - current["timestamp"]
        within_gap = time_gap <= max_gap_seconds

        if same_sender and within_gap:
            # Append with a space or newline based on content
            sep = "\n" if current["content"].endswith(("。", "！", "？", "~", "~", "…")) else " "
            current["content"] = current["content"] + sep + msg["content"]
            current["timestamp"] = msg["timestamp"]  # Use latest timestamp
        else:
            merged.append(current)
            current = msg.copy()

    merged.append(current)
    return merged


def clean_content(text: str) -> str:
    """Normalize message text."""
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize multiple spaces
    text = re.sub(r"  +", " ", text)
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text


def process_conversation(
    raw_messages: list[dict],
    target_wxid: str,
    max_gap_seconds: int,
) -> list[dict]:
    """
    Filter, clean, and merge messages for one conversation.
    Returns list of cleaned turn dicts.
    """
    # Filter to usable messages
    usable = [m for m in raw_messages if is_usable_message(m)]

    if not usable:
        return []

    # Clean content
    for m in usable:
        m["content"] = clean_content(m["content"])

    # Merge consecutive turns
    merged = merge_consecutive_turns(usable, max_gap_seconds)

    # Re-filter after merge (in case content got very short after cleaning)
    merged = [m for m in merged if len(m["content"]) >= 1]

    return merged


def main():
    parser = argparse.ArgumentParser(description="Clean WeChat messages")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    person_cfg = cfg["person"]
    data_cfg = cfg["data"]

    target_wxid = person_cfg.get("wxid", "")
    max_gap = data_cfg.get("max_gap_seconds", 3600)

    raw_dir = Path(data_cfg["raw_dir"])
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load contact name map
    contacts_file = raw_dir / "contacts.json"
    contact_names: dict[str, str] = {}
    if contacts_file.exists():
        with open(contacts_file) as f:
            contact_names = json.load(f)

    raw_files = list(raw_dir.glob("*.json"))
    raw_files = [f for f in raw_files if f.name != "contacts.json"]

    stats = {"conversations": 0, "total_messages": 0, "target_messages": 0, "skipped": 0}

    print(f"Processing {len(raw_files)} conversation files...")

    for raw_file in tqdm(raw_files):
        with open(raw_file, encoding="utf-8") as f:
            raw_messages = json.load(f)

        if not raw_messages:
            continue

        talker = raw_messages[0]["talker"]
        is_group = talker.endswith("@chatroom")

        cleaned = process_conversation(raw_messages, target_wxid, max_gap)

        if not cleaned:
            stats["skipped"] += 1
            continue

        # Check that the target person has at least some messages here
        target_turns = [m for m in cleaned if m["sender"] == target_wxid]
        if not target_turns:
            stats["skipped"] += 1
            continue

        # Build output structure
        output = {
            "talker": talker,
            "is_group": is_group,
            "contact_name": contact_names.get(talker, talker),
            "messages": cleaned,
        }

        out_path = out_dir / raw_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        stats["conversations"] += 1
        stats["total_messages"] += len(cleaned)
        stats["target_messages"] += len(target_turns)

    print(f"\nResults:")
    print(f"  Conversations kept: {stats['conversations']}")
    print(f"  Conversations skipped (no target messages): {stats['skipped']}")
    print(f"  Total turns: {stats['total_messages']}")
    print(f"  Target person turns: {stats['target_messages']}")
    print(f"  Output: {out_dir}")
    print(f"\nNext step: python scripts/03_build_training_data.py --config {args.config}")


if __name__ == "__main__":
    main()
