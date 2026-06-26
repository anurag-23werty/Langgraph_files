import gradio as gr
import uuid
from langchain_core.messages import HumanMessage
from langraph_backend import chatbot


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def respond(user_message: str, chat_history: list, thread_id: str):
    """Streaming generator — yields updated chat_history on every token chunk."""
    if not user_message.strip():
        yield chat_history, "", thread_id
        return

    # 1. Append user message immediately
    chat_history = chat_history + [{"role": "user", "content": user_message}]
    # 2. Add an empty assistant bubble that we'll fill token-by-token
    chat_history = chat_history + [{"role": "assistant", "content": ""}]
    yield chat_history, "", thread_id

    # 3. Stream from LangGraph — stream_mode="messages" gives (chunk, metadata) pairs
    accumulated = ""
    for message_chunk, _ in chatbot.stream(
        {"messages": [HumanMessage(content=user_message)]},
        config=get_config(thread_id),
        stream_mode="messages",
    ):
        token = message_chunk.content
        if token:                          # skip empty/tool chunks
            accumulated += token
            chat_history[-1] = {"role": "assistant", "content": accumulated}
            yield chat_history, "", thread_id


def clear_chat():
    return [], "", str(uuid.uuid4())


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
:root {
    --bg:     #212121;
    --bg2:    #2f2f2f;
    --accent: #19c37d;
    --fg:     #ececec;
    --muted:  #8e8ea0;
    --r:      12px;
}
body, .gradio-container { background: var(--bg) !important; color: var(--fg) !important; }
footer { display: none !important; }

#hdr { text-align:center; padding:24px 0 12px; border-bottom:1px solid #333; margin-bottom:8px; }
#hdr h1 { font-size:1.4rem; font-weight:600; margin:0; color:var(--fg); }
#hdr p  { font-size:0.78rem; color:var(--muted); margin:4px 0 0; }

#chatbox { background:var(--bg) !important; border:none !important; }

#msg textarea {
    background: var(--bg2) !important;
    border: 1px solid #444 !important;
    border-radius: var(--r) !important;
    color: var(--fg) !important;
    font-size: 0.95rem !important;
}
#msg textarea:focus { border-color: var(--accent) !important; box-shadow: none !important; }
#msg textarea::placeholder { color: var(--muted) !important; }

#send { background: var(--accent) !important; color: #111 !important;
        border-radius: var(--r) !important; font-weight:600 !important; border:none !important; }
#send:hover { filter: brightness(1.1); }

#clr { background: transparent !important; color: var(--muted) !important;
       border: 1px solid #444 !important; border-radius: var(--r) !important; }
#clr:hover { color: var(--fg) !important; border-color: #888 !important; }

#note { text-align:center; color:var(--muted); font-size:0.7rem; padding:8px 0 16px; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MistralAI Chat", fill_height=True) as demo:

    thread_id_state = gr.State(value=str(uuid.uuid4()))

    gr.HTML('<div id="hdr"><h1>✦ MistralAI Chat</h1><p>LangGraph · open-mistral-7b · streaming</p></div>')

    chatbot_ui = gr.Chatbot(
        value=[],
        elem_id="chatbox",
        label="",
        show_label=False,
        height=500,
        layout="bubble",
        render_markdown=True,
    )

    with gr.Row():
        msg = gr.Textbox(
            elem_id="msg",
            placeholder="Message MistralAI…",
            lines=1,
            max_lines=6,
            show_label=False,
            scale=9,
            container=False,
            autofocus=True,
            submit_btn=False,
        )
        send_btn = gr.Button("Send ↑", elem_id="send", scale=1, variant="primary", size="lg")

    with gr.Row():
        clear_btn = gr.Button("🗑 New chat", elem_id="clr", variant="secondary", size="sm")

    gr.HTML('<div id="note">Streaming via LangGraph stream_mode="messages" · memory per session</div>')

    # ── Events — note outputs include thread_id_state so the state round-trips ──
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot_ui, thread_id_state],
        outputs=[chatbot_ui, msg, thread_id_state],
    )
    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot_ui, thread_id_state],
        outputs=[chatbot_ui, msg, thread_id_state],
    )
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot_ui, msg, thread_id_state],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Base(),
    )