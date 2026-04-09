#!/usr/bin/env python3
"""
Memorial Chatbot — Gradio Web Interface

Serves the fine-tuned model with RAG memory retrieval via a simple web UI.
Friends can select their name and chat naturally.

Usage:
    python serving/app.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import yaml

from rag.prompt_builder import build_system_prompt
from rag.retriever import retrieve
from serving.ollama_client import chat_stream, is_model_available


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_app(cfg: dict):
    person_cfg = cfg["person"]
    rag_cfg = cfg["rag"]
    data_cfg = cfg["data"]
    serving_cfg = cfg["serving"]

    person_name = person_cfg.get("name", "friend")
    person_description = person_cfg.get("description", "")
    model_name = serving_cfg.get("ollama_model_name", "memorial-bot")
    contacts = serving_cfg.get("contacts", ["我自己"])
    chroma_dir = data_cfg["chroma_db_dir"]
    collection_name = rag_cfg["collection_name"]
    top_k = rag_cfg.get("top_k", 5)

    # Check Ollama model availability
    if not is_model_available(model_name):
        print(f"WARNING: Ollama model '{model_name}' not found.")
        print(f"  Run: ollama create {model_name} -f serving/Modelfile")

    def respond(user_message: str, history: list, contact_name: str):
        if not user_message.strip():
            return "", history

        # Retrieve relevant memories
        memories = []
        try:
            memories = retrieve(
                query=user_message,
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                top_k=top_k,
                contact_filter=contact_name if contact_name != "我自己" else None,
            )
        except Exception as e:
            print(f"RAG retrieval error (non-fatal): {e}")

        system_prompt = build_system_prompt(
            person_name=person_name,
            contact_name=contact_name,
            memories=memories,
            person_description=person_description,
        )

        # Stream response
        full_response = ""
        history = history + [(user_message, "")]

        for chunk in chat_stream(
            model_name=model_name,
            system_prompt=system_prompt,
            history=history[:-1],  # Exclude the current incomplete turn
            user_message=user_message,
        ):
            full_response += chunk
            history[-1] = (user_message, full_response)
            yield "", history

    def clear_chat():
        return [], ""

    # Build UI
    with gr.Blocks(
        title=f"与{person_name}聊天",
        theme=gr.themes.Soft(),
        css="""
        .disclaimer {
            font-size: 0.8em;
            color: #888;
            text-align: center;
            margin-top: 8px;
        }
        """,
    ) as demo:
        gr.Markdown(f"# 与{person_name}聊天")
        gr.Markdown(
            "这是一个用来怀念朋友的聊天机器人。"
            "它学习了他的说话方式和记忆，"
            "让我们可以继续和他聊天。",
        )
        gr.HTML(
            '<p class="disclaimer">'
            "注意：这是一个AI模拟，不是真实的人。请以思念和温情的心情来使用。"
            "</p>"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label=person_name,
                    height=500,
                    bubble_full_width=False,
                    show_label=True,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder=f"和{person_name}说点什么...",
                        show_label=False,
                        scale=9,
                        lines=1,
                        max_lines=5,
                    )
                    send_btn = gr.Button("发送", scale=1, variant="primary")

            with gr.Column(scale=1):
                contact_dropdown = gr.Dropdown(
                    choices=contacts,
                    value=contacts[0] if contacts else "我自己",
                    label="你是谁？",
                    info="选择你的名字，让他认出你",
                )
                clear_btn = gr.Button("清空对话", variant="secondary")

                gr.Markdown("### 使用说明")
                gr.Markdown(
                    "1. 选择你的名字\n"
                    "2. 在输入框里打字\n"
                    "3. 按回车或点发送\n\n"
                    "机器人会记住他说过的话和经历，尽量用他的语气来回复。"
                )

        # Wire up events
        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, contact_dropdown],
            outputs=[msg_box, chatbot],
        )
        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, contact_dropdown],
            outputs=[msg_box, chatbot],
        )
        clear_btn.click(clear_chat, outputs=[chatbot, msg_box])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Start memorial chatbot UI")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    serving_cfg = cfg["serving"]

    demo = create_app(cfg)
    demo.launch(
        server_name=serving_cfg.get("host", "0.0.0.0"),
        server_port=serving_cfg.get("port", 7860),
        share=False,
    )


if __name__ == "__main__":
    main()
