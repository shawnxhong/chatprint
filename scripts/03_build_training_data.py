#!/usr/bin/env python3
"""
Script 03: Build ShareGPT training pairs for QLoRA fine-tuning

Reads cleaned conversation files from 02_clean_data.py and builds
multi-turn conversation pairs in ShareGPT format where the target
person's messages are the "assistant" responses.

Usage:
    python scripts/03_build_training_data.py --config config.yaml
"""

import argparse
import json
import random
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_system_prompt(person_name: str, contact_name: str, person_description: str = "") -> str:
    desc = f"\n{person_description}" if person_description else ""
    return (
        f"你是{person_name}，正在和{contact_name}聊天。{desc}"
        f"请用{person_name}平时的语气、口头禅和说话风格来回复，不要太正式，就像真实聊天一样。"
    )


def turns_to_sharegpt(
    turns: list[dict],
    target_wxid: str,
    person_name: str,
    contact_name: str,
    person_description: str,
    context_min: int,
    context_max: int,
    min_response_length: int,
) -> list[dict]:
    """
    Convert a sequence of conversation turns into ShareGPT examples.

    For each turn by the target person, we use the preceding turns as
    the conversation context (human side) and the target's message as
    the assistant response.
    """
    examples = []

    for i, turn in enumerate(turns):
        if turn["sender"] != target_wxid:
            continue

        response = turn["content"].strip()
        if len(response) < min_response_length:
            continue

        # Pick a context window ending just before this turn
        context_size = random.randint(context_min, context_max)
        context_turns = turns[max(0, i - context_size):i]

        if not context_turns:
            continue

        # Build the conversation in ShareGPT format
        system_prompt = build_system_prompt(person_name, contact_name, person_description)
        conversation = [{"from": "system", "value": system_prompt}]

        # Interleave context turns as human/gpt exchanges
        # We alternate: non-target -> human, target -> gpt
        # But WeChat conversations can have multiple people; we simplify to
        # "everything that's not the target" = human side
        for ctx_turn in context_turns:
            sender_name = ctx_turn.get("contact_name") or ctx_turn["sender"]
            content = ctx_turn["content"].strip()
            if not content:
                continue

            if ctx_turn["sender"] == target_wxid:
                conversation.append({"from": "gpt", "value": content})
            else:
                # Label with sender name so the model knows who it's responding to
                labeled = f"{sender_name}: {content}"
                conversation.append({"from": "human", "value": labeled})

        # The last turn in context must be from human (otherwise skip — the model
        # shouldn't start a new response if it just responded)
        if not conversation or conversation[-1]["from"] != "human":
            continue

        # Add the target's response
        conversation.append({"from": "gpt", "value": response})

        examples.append({"conversations": conversation})

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build training data for QLoRA")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    cfg = load_config(args.config)
    person_cfg = cfg["person"]
    data_cfg = cfg["data"]

    target_wxid = person_cfg.get("wxid", "")
    person_name = person_cfg.get("name", "friend")
    person_description = person_cfg.get("description", "")
    priority_contacts = cfg["wechat"].get("priority_contacts", [])

    context_min = data_cfg.get("context_window_min", 3)
    context_max = data_cfg.get("context_window_max", 8)
    max_gap = data_cfg.get("max_gap_seconds", 3600)
    min_response_len = data_cfg.get("min_response_length", 2)
    max_examples = data_cfg.get("max_training_examples", 10000)

    processed_dir = Path(data_cfg["processed_dir"])
    out_dir = Path(data_cfg["training_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load contact names map for labeling context turns
    contact_names: dict[str, str] = {}
    contacts_file = Path(data_cfg["raw_dir"]) / "contacts.json"
    if contacts_file.exists():
        with open(contacts_file) as f:
            contact_names = json.load(f)

    conv_files = list(processed_dir.glob("*.json"))

    # Sort: priority contacts first (1-on-1), then groups last
    def sort_key(f):
        with open(f, encoding="utf-8") as fp:
            meta = json.load(fp)
        talker = meta.get("talker", "")
        is_group = meta.get("is_group", False)
        is_priority = talker in priority_contacts
        return (0 if is_priority else (1 if not is_group else 2))

    conv_files.sort(key=sort_key)

    all_examples: list[dict] = []
    stats = {"files": 0, "direct": 0, "group": 0}

    print(f"Building training pairs from {len(conv_files)} conversations...")

    for conv_file in tqdm(conv_files):
        if len(all_examples) >= max_examples:
            break

        with open(conv_file, encoding="utf-8") as f:
            conv = json.load(f)

        talker = conv.get("talker", "")
        is_group = conv.get("is_group", False)
        contact_name = conv.get("contact_name") or contact_names.get(talker, talker)
        messages = conv.get("messages", [])

        # Enrich turns with contact name for labeling
        for m in messages:
            if m["sender"] != target_wxid:
                m["contact_name"] = contact_names.get(m["sender"], m["sender"])

        # Split into sub-conversations at time gaps
        sub_convs: list[list[dict]] = []
        current_sub: list[dict] = []
        for i, msg in enumerate(messages):
            if i > 0 and (msg["timestamp"] - messages[i - 1]["timestamp"]) > max_gap * 2:
                if current_sub:
                    sub_convs.append(current_sub)
                current_sub = [msg]
            else:
                current_sub.append(msg)
        if current_sub:
            sub_convs.append(current_sub)

        file_examples = []
        for sub in sub_convs:
            examples = turns_to_sharegpt(
                turns=sub,
                target_wxid=target_wxid,
                person_name=person_name,
                contact_name=contact_name,
                person_description=person_description,
                context_min=context_min,
                context_max=context_max,
                min_response_length=min_response_len,
            )
            file_examples.extend(examples)

        all_examples.extend(file_examples)
        stats["files"] += 1
        if is_group:
            stats["group"] += len(file_examples)
        else:
            stats["direct"] += len(file_examples)

    # Shuffle and cap
    random.shuffle(all_examples)
    all_examples = all_examples[:max_examples]

    # Split train/val
    val_cfg = cfg.get("training", {}).get("val_split", 0.05)
    val_size = max(1, int(len(all_examples) * val_cfg))
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]

    train_path = out_dir / "train.json"
    val_path = out_dir / "val.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_examples, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_examples, f, ensure_ascii=False, indent=2)

    print(f"\nResults:")
    print(f"  Total examples: {len(all_examples)}")
    print(f"  Train: {len(train_examples)} | Val: {len(val_examples)}")
    print(f"  From 1-on-1 chats: {stats['direct']} | From group chats: {stats['group']}")
    print(f"  Output: {out_dir}")
    print(f"\nNext step: python scripts/04_build_rag_corpus.py --config {args.config}")


if __name__ == "__main__":
    main()
