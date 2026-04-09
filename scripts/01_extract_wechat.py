#!/usr/bin/env python3
"""
Script 01: WeChat Data Extraction

Decrypts and exports WeChat chat history using PyWxDump.

Usage:
    python scripts/01_extract_wechat.py --config config.yaml

Requirements:
    - Run on the machine where WeChat is (or was) installed
    - WeChat must have been opened at least once (to generate the key)
    - On Windows: WeChat process should be running for auto key extraction,
      or provide the key manually via --key
    - On Mac: use wechatDataBackup GUI tool first, then point --data-dir to the output
"""

import argparse
import json
import os
import sqlite3
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_group_sender(content: str) -> str | None:
    """Extract sender wxid from group chat message XML content."""
    try:
        root = ET.fromstring(content)
        sender = root.find(".//fromusername")
        if sender is not None:
            return sender.text
    except ET.ParseError:
        pass
    # Fallback: some group messages prefix with "wxid_xxx:\n"
    if ":\n" in content:
        possible_wxid = content.split(":\n", 1)[0].strip()
        if possible_wxid and " " not in possible_wxid and len(possible_wxid) < 50:
            return possible_wxid
    return None


def export_from_decrypted_db(db_path: str, output_dir: Path, wxid: str) -> int:
    """
    Export messages from an already-decrypted WeChat MSG database.
    Returns number of messages exported.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Check which tables exist
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur.fetchall()}

    if "MSG" not in tables:
        print(f"  No MSG table in {db_path}, skipping.")
        conn.close()
        return 0

    cur.execute("""
        SELECT MsgSvrID, StrTalker, StrContent, IsSender, Type, SubType, CreateTime, MsgSeq
        FROM MSG
        ORDER BY CreateTime ASC
    """)

    rows = cur.fetchall()
    conn.close()

    # Group by conversation (StrTalker)
    conversations: dict[str, list] = {}
    for row in rows:
        talker = row["StrTalker"]
        msg_type = row["Type"]
        content = row["StrContent"] or ""
        is_sender = row["IsSender"]

        # Determine real sender for group chats
        if talker.endswith("@chatroom"):
            if is_sender:
                sender_wxid = wxid
            else:
                sender_wxid = parse_group_sender(content) or "unknown"
            # Strip XML wrapper to get plain text
            if content.startswith("<"):
                try:
                    root = ET.fromstring(content)
                    text_node = root.find(".//content")
                    if text_node is not None and text_node.text:
                        content = text_node.text.strip()
                    else:
                        # Try getting text after ":\n" prefix
                        if ":\n" in content:
                            content = content.split(":\n", 1)[1].strip()
                except ET.ParseError:
                    if ":\n" in content:
                        content = content.split(":\n", 1)[1].strip()
        else:
            sender_wxid = wxid if is_sender else talker

        msg = {
            "id": row["MsgSvrID"],
            "talker": talker,
            "sender": sender_wxid,
            "content": content,
            "is_sender": bool(is_sender),
            "type": msg_type,
            "timestamp": row["CreateTime"],
            "seq": row["MsgSeq"],
        }

        if talker not in conversations:
            conversations[talker] = []
        conversations[talker].append(msg)

    # Write one JSON file per conversation
    count = 0
    for talker, messages in conversations.items():
        safe_name = talker.replace("@", "_").replace("/", "_")
        out_path = output_dir / f"{safe_name}.json"
        # Append if file already exists (multiple MSG databases)
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            existing.extend(messages)
            # Re-sort by timestamp
            existing.sort(key=lambda m: m["timestamp"])
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
        count += len(messages)

    return count


def export_contacts(db_path: str, output_dir: Path) -> dict:
    """Export contact list to map wxid -> display name."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur.fetchall()}

    contacts = {}
    if "Contact" in tables:
        cur.execute("SELECT UserName, NickName, Remark FROM Contact")
        for row in cur.fetchall():
            wxid = row["UserName"]
            # Prefer Remark (user-set nickname) over NickName
            name = row["Remark"] or row["NickName"] or wxid
            contacts[wxid] = name

    conn.close()

    out_path = output_dir / "contacts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contacts, f, ensure_ascii=False, indent=2)
    print(f"  Exported {len(contacts)} contacts to {out_path}")
    return contacts


def extract_via_pywxdump(data_dir: str, output_dir: Path, wxid: str):
    """
    Use PyWxDump to decrypt WeChat databases and export messages.
    Requires PyWxDump installed and WeChat data directory accessible.
    """
    try:
        from pywxdump import WxInfo
        from pywxdump.db import DBHandler
    except ImportError:
        print("ERROR: pywxdump not installed. Run: pip install pywxdump")
        sys.exit(1)

    print("Attempting auto key extraction via PyWxDump...")
    wx_infos = WxInfo.get_wx_infos()
    if not wx_infos:
        print("ERROR: Could not auto-detect WeChat process or key.")
        print("Make sure WeChat is running, or use --key to provide the key manually.")
        sys.exit(1)

    # Use the first account found (or match wxid if provided)
    info = wx_infos[0]
    if wxid:
        for i in wx_infos:
            if i.get("wxid") == wxid:
                info = i
                break

    key = info.get("key")
    db_dir = info.get("db_dir") or data_dir
    print(f"Found WeChat account: {info.get('wxid')}, db_dir: {db_dir}")

    # Decrypt all MSG databases
    decrypted_dir = output_dir / "decrypted"
    decrypted_dir.mkdir(exist_ok=True)

    db_files = list(Path(db_dir).glob("MSG*.db"))
    if not db_files:
        print(f"ERROR: No MSG*.db files found in {db_dir}")
        sys.exit(1)

    print(f"Found {len(db_files)} MSG database(s).")

    handler = DBHandler(db_dir=str(db_dir), key=key, out_dir=str(decrypted_dir))
    handler.decrypt()
    print("Databases decrypted.")

    return decrypted_dir


def main():
    parser = argparse.ArgumentParser(description="Extract WeChat chat history")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data-dir", help="Override WeChat data directory from config")
    parser.add_argument("--key", help="WeChat database encryption key (hex string)")
    parser.add_argument(
        "--decrypted-dir",
        help="Skip decryption; point to directory of already-decrypted .db files",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    person_cfg = cfg["person"]
    wechat_cfg = cfg["wechat"]
    data_cfg = cfg["data"]

    wxid = person_cfg.get("wxid", "")
    data_dir = args.data_dir or wechat_cfg.get("data_dir", "")
    output_dir = Path(data_cfg["raw_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.decrypted_dir:
        decrypted_dir = Path(args.decrypted_dir)
        print(f"Using pre-decrypted databases from: {decrypted_dir}")
    else:
        if not data_dir:
            print("ERROR: Set wechat.data_dir in config.yaml or use --data-dir")
            sys.exit(1)
        decrypted_dir = extract_via_pywxdump(data_dir, output_dir, wxid)

    # Export contacts first
    contact_db = decrypted_dir / "MicroMsg.db"
    if contact_db.exists():
        export_contacts(str(contact_db), output_dir)
    else:
        print("WARNING: MicroMsg.db not found; contact names will use wxids")

    # Export messages from all MSG databases
    total = 0
    for db_file in sorted(decrypted_dir.glob("MSG*.db")):
        print(f"Processing {db_file.name}...")
        n = export_from_decrypted_db(str(db_file), output_dir, wxid)
        print(f"  Exported {n} messages")
        total += n

    print(f"\nDone. Total messages exported: {total}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext step: python scripts/02_clean_data.py --config {args.config}")


if __name__ == "__main__":
    main()
