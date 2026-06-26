import gradio as gr
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langraph_backend import chatbot  # single instance, InMemorySaver lives here


# ── Utilities ─────────────────────────────────────────────────────────────────
def generate_thread_id() -> str:
    return str(uuid.uuid4())

def get_config(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}}

def load_conversation(tid: str) -> list:
    """Pull full history from LangGraph checkpointer."""
    try:
        state = chatbot.get_state(config=get_config(tid))
        history = []
        for msg in state.values.get("messages", []):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            if msg.content:
                history.append({"role": role, "content": msg.content})
        return history
    except Exception as e:
        print(f"[load_conversation] error for {tid}: {e}")
        return []

def short_label(tid: str, idx: int) -> str:
    return f"Chat {idx + 1}"

def build_sidebar_buttons(threads: list, active_id: str):
    """Return one gr.Button update per slot (we pre-create 20 slots)."""
    updates = []
    for i in range(MAX_THREADS):
        if i < len(threads):
            tid = threads[i]
            label = short_label(tid, i)
            prefix = "▶ " if tid == active_id else "   "
            updates.append(gr.update(value=prefix + label, visible=True, variant="secondary"))
        else:
            updates.append(gr.update(visible=False))
    return updates


MAX_THREADS = 20   # max sidebar slots


# ── Core functions ─────────────────────────────────────────────────────────────
def respond(user_message, chat_history, tid, threads):
    if not user_message.strip():
        yield [chat_history, "", tid, threads] + build_sidebar_buttons(threads, tid)
        return
    if tid not in threads:
        threads = threads + [tid]
    chat_history = chat_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": ""},
    ]
    yield [chat_history, "", tid, threads] + build_sidebar_buttons(threads, tid)

    accumulated = ""
    for chunk, _ in chatbot.stream(
        {"messages": [HumanMessage(content=user_message)]},
        config=get_config(tid),
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage) and chunk.content:
            accumulated += chunk.content
            chat_history[-1] = {"role": "assistant", "content": accumulated}
            # Only update chatbot during streaming, NOT the sidebar buttons
            yield [chat_history, "", tid, threads] + [gr.update()] * MAX_THREADS


def new_chat(threads, current_tid):
    if current_tid and current_tid not in threads:
        threads = threads + [current_tid]
    new_tid = generate_thread_id()
    threads = threads + [new_tid]
    return [[], "", new_tid, threads] + build_sidebar_buttons(threads, new_tid)


def make_switch_fn(slot_index: int):
    """Factory: returns a click handler for sidebar slot i."""
    def switch(threads, current_tid):
        if slot_index >= len(threads):
            return [gr.update(), "", current_tid, threads] + build_sidebar_buttons(threads, current_tid)
        selected_tid = threads[slot_index]
        if current_tid and current_tid not in threads:
            threads = threads + [current_tid]
        history = load_conversation(selected_tid)
        print(f"[switch] slot={slot_index} tid={selected_tid} messages={len(history)}")
        return [history, "", selected_tid, threads] + build_sidebar_buttons(threads, selected_tid)
    return switch


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
:root { --bg:#212121; --sb:#171717; --bg2:#2f2f2f; --accent:#19c37d; --fg:#ececec; --muted:#8e8ea0; --r:10px; }
body, .gradio-container { background:var(--bg) !important; color:var(--fg) !important; font-family:'Inter',system-ui,sans-serif !important; }
footer { display:none !important; }

#sidebar-col { background:var(--sb) !important; border-right:1px solid #2a2a2a; padding:16px 12px !important; min-height:100vh; }
#main-col    { padding:0 16px !important; }

#hdr { text-align:center; padding:20px 0 10px; border-bottom:1px solid #333; margin-bottom:10px; }
#hdr h1 { font-size:1.3rem; font-weight:600; margin:0; color:var(--fg); }
#hdr p  { font-size:0.75rem; color:var(--muted); margin:4px 0 0; }

/* new chat button */
#new-btn { background:var(--accent) !important; color:#111 !important; border-radius:var(--r) !important;
           font-weight:600 !important; border:none !important; width:100% !important; margin-bottom:14px !important; }
#new-btn:hover { filter:brightness(1.1); }

/* thread slot buttons */
.thread-slot button {
    background: transparent !important;
    color: var(--fg) !important;
    border: none !important;
    border-radius: 8px !important;
    text-align: left !important;
    font-size: 0.78rem !important;
    padding: 9px 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    margin-bottom: 2px !important;
    transition: background 0.15s;
}
.thread-slot button:hover { background: #2a2a2a !important; }

#chatbox { background:var(--bg) !important; border:none !important; }

#msg textarea { background:var(--bg2) !important; border:1px solid #444 !important;
                border-radius:var(--r) !important; color:var(--fg) !important; font-size:0.95rem !important; }
#msg textarea:focus { border-color:var(--accent) !important; box-shadow:none !important; }
#msg textarea::placeholder { color:var(--muted) !important; }

#send { background:var(--accent) !important; color:#111 !important; border-radius:var(--r) !important;
        font-weight:600 !important; border:none !important; }
#send:hover { filter:brightness(1.1); }
#note { text-align:center; color:var(--muted); font-size:0.68rem; padding:6px 0 12px; }
"""


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MistralAI Chat", fill_height=True) as demo:

    init_tid     = generate_thread_id()
    thread_id_st = gr.State(value=init_tid)
    threads_st   = gr.State(value=[init_tid])

    # outputs: chatbot, msg, thread_id_st, threads_st, + MAX_THREADS button updates
    all_outs_refs = []   # filled after buttons are created

    with gr.Row():
        # ── Sidebar ──────────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="sidebar-col"):
            gr.HTML(
                '<p style="font-size:1rem;font-weight:600;color:#ececec;margin:0 0 2px;">✦ MistralAI</p>'
                '<p style="font-size:0.72rem;color:#8e8ea0;margin:0 0 14px;">LangGraph · open-mistral-7b</p>'
            )
            new_btn = gr.Button("+ New Chat", elem_id="new-btn", variant="primary")
            gr.HTML('<p style="font-size:0.72rem;color:#8e8ea0;margin:0 0 6px;">My Conversations</p>')

            # Pre-create MAX_THREADS real Gradio buttons — no JS needed
            slot_buttons = []
            for i in range(MAX_THREADS):
                visible = (i == 0)   # only first slot visible initially
                label   = short_label(init_tid, i) if i == 0 else f"Chat {i+1}"
                b = gr.Button(
                    value=("▶ " if i == 0 else "   ") + label,
                    visible=visible,
                    elem_classes=["thread-slot"],
                )
                slot_buttons.append(b)

        # ── Main ─────────────────────────────────────────────────────────────
        with gr.Column(scale=4, elem_id="main-col"):
            gr.HTML(
                '<div id="hdr"><h1>MistralAI Chat</h1>'
                '<p>Streaming · multi-conversation · per-session memory</p></div>'
            )
            chatbot_ui = gr.Chatbot(
                value=[], elem_id="chatbox", label="", show_label=False,
                height=500, layout="bubble", render_markdown=True,
            )
            with gr.Row():
                msg = gr.Textbox(
                    elem_id="msg", placeholder="Message MistralAI…",
                    lines=1, max_lines=6, show_label=False,
                    scale=9, container=False, autofocus=True, submit_btn=False,
                )
                send_btn = gr.Button("Send ↑", elem_id="send", scale=1, variant="primary", size="lg")
            gr.HTML('<div id="note">Streaming via stream_mode="messages" · memory via InMemorySaver</div>')

    # ── Wire events ───────────────────────────────────────────────────────────
    # Outputs = [chatbot, msg_textbox, thread_id_state, threads_state, btn0, btn1, ..., btnN]
    base_outs  = [chatbot_ui, msg, thread_id_st, threads_st]
    all_outs   = base_outs + slot_buttons

    msg.submit(respond,     [msg, chatbot_ui, thread_id_st, threads_st], all_outs)
    send_btn.click(respond, [msg, chatbot_ui, thread_id_st, threads_st], all_outs)
    new_btn.click(new_chat, [threads_st, thread_id_st], all_outs)

    # Each slot button gets its own dedicated handler — no JS, no bridge, pure Gradio
    for i, btn in enumerate(slot_buttons):
        btn.click(
            fn=make_switch_fn(i),
            inputs=[threads_st, thread_id_st],
            outputs=all_outs,
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CSS,
        theme=gr.themes.Base(),
    )