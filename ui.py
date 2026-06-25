import gradio as gr
from langchain_core.messages import HumanMessage
from langraph_backend import chatbot

# ── Persistent thread config (one per Gradio session via state) ──────────────
def get_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def respond(user_message: str, chat_history: list, thread_id: str):
    """
    Called on every user submission.
    chat_history: list of (user_str, assistant_str) tuples — Gradio's native format.
    thread_id:    kept in gr.State so each browser tab gets its own memory.
    """
    if not user_message.strip():
        return chat_history, ""

    config = get_config(thread_id)

    # Invoke LangGraph — checkpointer replays full thread history automatically
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    ai_reply = response["messages"][-1].content

    # Append to Gradio history
    chat_history = chat_history + [(user_message, ai_reply)]
    return chat_history, ""          # second value clears the input box


def clear_chat(thread_id: str):
    """Reset UI history; bump thread_id so LangGraph starts a fresh memory."""
    import uuid
    new_thread_id = str(uuid.uuid4())
    return [], "", new_thread_id


# ── CSS — dark theme, ChatGPT-like proportions ───────────────────────────────
CSS = """
/* ── root variables ── */
:root {
    --bg-main:    #212121;
    --bg-sidebar: #171717;
    --bg-input:   #2f2f2f;
    --bg-user:    #2f2f2f;
    --accent:     #19c37d;
    --text-main:  #ececec;
    --text-muted: #8e8ea0;
    --radius:     12px;
    --font:       'Inter', system-ui, sans-serif;
}

/* ── page shell ── */
body, .gradio-container {
    background: var(--bg-main) !important;
    color: var(--text-main) !important;
    font-family: var(--font) !important;
}

/* ── header ── */
#header {
    text-align: center;
    padding: 28px 0 8px;
    border-bottom: 1px solid #333;
}
#header h1 {
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text-main);
    margin: 0;
    letter-spacing: -0.3px;
}
#header p {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 4px 0 0;
}

/* ── chatbox ── */
#chatbox {
    background: var(--bg-main) !important;
    border: none !important;
    padding: 12px 0 !important;
}

/* user bubble */
#chatbox .message.user {
    background: var(--bg-user) !important;
    color: var(--text-main) !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important;
    max-width: 78% !important;
    margin-left: auto !important;
    font-size: 0.93rem;
    line-height: 1.55;
}

/* assistant bubble */
#chatbox .message.bot {
    background: transparent !important;
    color: var(--text-main) !important;
    border-radius: var(--radius) !important;
    padding: 12px 16px !important;
    max-width: 86% !important;
    font-size: 0.93rem;
    line-height: 1.65;
    border-left: 2px solid var(--accent) !important;
}

/* ── input row ── */
#input-row {
    background: var(--bg-input) !important;
    border-radius: 14px !important;
    border: 1px solid #444 !important;
    display: flex;
    align-items: center;
    padding: 4px 8px;
    margin: 12px 0;
}

#user-input textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-main) !important;
    font-size: 0.95rem !important;
    resize: none !important;
    box-shadow: none !important;
    outline: none !important;
}
#user-input textarea::placeholder { color: var(--text-muted) !important; }
#user-input textarea:focus { box-shadow: none !important; border: none !important; }

/* ── buttons ── */
#send-btn {
    background: var(--accent) !important;
    color: #111 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 18px !important;
    cursor: pointer !important;
    min-width: 72px;
}
#send-btn:hover { filter: brightness(1.1); }

#clear-btn {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid #444 !important;
    border-radius: 10px !important;
    font-size: 0.8rem !important;
    padding: 7px 14px !important;
    cursor: pointer !important;
}
#clear-btn:hover { color: var(--text-main) !important; border-color: #888 !important; }

/* ── footer note ── */
#footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.72rem;
    padding: 6px 0 18px;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="MistralAI Chat", theme=gr.themes.Base()) as demo:

    # Hidden state — one thread_id per session tab
    import uuid
    thread_id_state = gr.State(value=str(uuid.uuid4()))

    # Header
    gr.HTML("""
        <div id="header">
            <h1>✦ MistralAI Chat</h1>
            <p>Powered by LangGraph · open-mistral-7b</p>
        </div>
    """)

    # Conversation display
    chatbot_ui = gr.Chatbot(
        elem_id="chatbox",
        label="",
        height=520,
        show_label=False,
        bubble_full_width=False,
        avatar_images=(None, "https://i.imgur.com/0YM3LjH.png"),  # user / bot
    )

    # Input row
    with gr.Row(elem_id="input-row"):
        user_input = gr.Textbox(
            elem_id="user-input",
            placeholder="Message MistralAI...",
            lines=1,
            max_lines=6,
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send ↑", elem_id="send-btn", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑  New chat", elem_id="clear-btn", scale=1)

    gr.HTML('<div id="footer">Memory is preserved within a session via LangGraph InMemorySaver.</div>')

    # ── Wire up events ────────────────────────────────────────────────────────

    # Submit on Enter (shift+enter = newline)
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot_ui, thread_id_state],
        outputs=[chatbot_ui, user_input],
    )

    # Submit on button click
    send_btn.click(
        fn=respond,
        inputs=[user_input, chatbot_ui, thread_id_state],
        outputs=[chatbot_ui, user_input],
    )

    # Clear resets UI and rotates the thread_id so LangGraph forgets history
    clear_btn.click(
        fn=clear_chat,
        inputs=[thread_id_state],
        outputs=[chatbot_ui, user_input, thread_id_state],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)