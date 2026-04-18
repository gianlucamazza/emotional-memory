"""Emotional Memory — Affective Field Theory live demo.

Runs as a Hugging Face Space (sdk: gradio) or locally::

    pip install "emotional-memory[sentence-transformers]" gradio matplotlib
    python demo/app.py
"""

from __future__ import annotations

import hashlib
import io

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emotional_memory import EmotionalMemory, InMemoryStore
from emotional_memory.integrations import EmotionalMemoryChatHistory

# ---------------------------------------------------------------------------
# Embedder — use SentenceTransformer if available, fall back to Hash
# ---------------------------------------------------------------------------


def _make_embedder():  # type: ignore[no-untyped-def]
    try:
        from emotional_memory.embedders import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    except ImportError:
        return _HashEmbedder()


class _HashEmbedder:
    def embed(self, text: str) -> list[float]:
        d = hashlib.sha256(text.encode()).digest()
        return [(b / 127.5) - 1.0 for b in d[:64]]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Global state (one instance per Gradio session via State)
# ---------------------------------------------------------------------------


def _build_engine() -> tuple[EmotionalMemory, EmotionalMemoryChatHistory]:
    em = EmotionalMemory(store=InMemoryStore(), embedder=_make_embedder())
    history = EmotionalMemoryChatHistory(em)
    return em, history


# ---------------------------------------------------------------------------
# PAD plot
# ---------------------------------------------------------------------------

_PAD_HISTORY_MAX = 40


def _pad_plot(pad_history: list[tuple[float, float, float]]) -> bytes:
    """Return a PNG bytes object of the PAD trajectory plot."""
    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    labels = ["Valence", "Arousal", "Dominance"]
    colors = ["#7ec8e3", "#f5a623", "#b8e986"]
    ylims = [(-1, 1), (0, 1), (-1, 1)]

    xs = list(range(len(pad_history)))
    for i, (ax, label, color, ylim) in enumerate(zip(axes, labels, colors, ylims, strict=False)):
        ys = [p[i] for p in pad_history]
        ax.set_facecolor("#1a1a2e")
        ax.plot(xs, ys, color=color, linewidth=1.5)
        if len(ys) > 1:
            ax.fill_between(xs, ys, alpha=0.15, color=color)
        ax.set_ylim(ylim)
        ax.set_ylabel(label, color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.axhline(0, color="#444", linewidth=0.5, linestyle="--")

    axes[-1].set_xlabel("Turn", color="white", fontsize=8)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------


def chat(
    user_msg: str,
    chat_history: list[dict[str, str]],
    em_state: EmotionalMemory,
    history_state: EmotionalMemoryChatHistory,
    pad_history: list[tuple[float, float, float]],
) -> tuple[
    list[dict[str, str]],
    EmotionalMemory,
    EmotionalMemoryChatHistory,
    list[tuple[float, float, float]],
    bytes,
    str,
]:
    if not user_msg.strip():
        plot = _pad_plot(pad_history) if pad_history else _pad_plot([(0.0, 0.5, 0.0)])
        return chat_history, em_state, history_state, pad_history, plot, ""

    history_state.add_user_message(user_msg)

    state = em_state.get_state()
    v = state.core_affect.valence
    a = state.core_affect.arousal

    # Simple rule-based reply that reflects the affective state
    if v > 0.3 and a > 0.5:
        tone = "enthusiastically"
    elif v > 0.1:
        tone = "warmly"
    elif v < -0.3 and a > 0.5:
        tone = "with concern"
    elif v < -0.1:
        tone = "gently"
    else:
        tone = "calmly"

    reply = (
        f"[{tone}] I heard you. "
        f"Current state — valence: {v:+.2f}, arousal: {a:.2f}. "
        f"I have {len(em_state.list_all())} memories stored. "
        "Ask me to 'recall' something to see mood-congruent retrieval."
    )

    # Handle recall requests
    if "recall" in user_msg.lower() or "remember" in user_msg.lower():
        results = em_state.retrieve(user_msg, top_k=3)
        if results:
            excerpts = "; ".join(f'"{m.content[:60]}…"' for m in results[:3])
            reply = f"[{tone}] Top memories matching your current mood: {excerpts}"
        else:
            reply = f"[{tone}] No memories yet — keep chatting to build them up!"

    history_state.add_ai_message(reply)

    state = em_state.get_state()
    mood = state.mood
    pad_history = [
        *pad_history,
        (state.core_affect.valence, state.core_affect.arousal, mood.dominance),
    ]
    if len(pad_history) > _PAD_HISTORY_MAX:
        pad_history = pad_history[-_PAD_HISTORY_MAX:]

    chat_history = [
        *chat_history,
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    plot = _pad_plot(pad_history)
    return chat_history, em_state, history_state, pad_history, plot, ""


def reset_session() -> tuple[
    list,
    EmotionalMemory,
    EmotionalMemoryChatHistory,
    list,
    bytes,
]:
    em, history = _build_engine()
    plot = _pad_plot([(0.0, 0.5, 0.0)])
    return [], em, history, [(0.0, 0.5, 0.0)], plot


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_DESCRIPTION = """
# 🧠 Emotional Memory — Affective Field Theory Demo

Every message you send is encoded with a full **affective fingerprint** (valence, arousal,
appraisal, mood, resonance links). The PAD plot shows your conversation's emotional trajectory
in real time. Type **"recall X"** to trigger mood-congruent retrieval.

Built with [`emotional-memory`](https://github.com/gianlucamazza/emotional-memory).
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Emotional Memory Demo") as demo:
    gr.Markdown(_DESCRIPTION)

    em_state = gr.State()
    history_state = gr.State()
    pad_history = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=420, type="messages")
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type a message… try expressing joy, fear, sadness, or calm.",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            reset_btn = gr.Button("🔄 New session", variant="secondary", size="sm")

        with gr.Column(scale=2):
            pad_plot = gr.Image(label="PAD state trajectory", type="pil")
            gr.Markdown(
                "**Valence** (blue): negative ↔ positive  \n"
                "**Arousal** (orange): calm ↔ excited  \n"
                "**Dominance** (green): submissive ↔ dominant"
            )

    # Init on load
    demo.load(
        fn=reset_session,
        outputs=[chatbot, em_state, history_state, pad_history, pad_plot],
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, em_state, history_state, pad_history],
        outputs=[chatbot, em_state, history_state, pad_history, pad_plot, msg_box],
    )
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, em_state, history_state, pad_history],
        outputs=[chatbot, em_state, history_state, pad_history, pad_plot, msg_box],
    )
    reset_btn.click(
        fn=reset_session,
        outputs=[chatbot, em_state, history_state, pad_history, pad_plot],
    )

if __name__ == "__main__":
    demo.launch()
