"""Emotional Memory — Affective Field Theory live demo.

Runs as a Hugging Face Space (sdk: gradio) or locally::

    pip install "emotional-memory[langchain]" httpx gradio matplotlib
    python demo/app.py

Set EMOTIONAL_MEMORY_LLM_API_KEY (+ optionally EMOTIONAL_MEMORY_LLM_MODEL and
EMOTIONAL_MEMORY_LLM_BASE_URL) to enable LLM-backed appraisal (multilingual,
full AFT pipeline).  Without it the demo falls back to KeywordAppraisalEngine
(rule-based, English only).
"""

from __future__ import annotations

import hashlib
import io
import os

import gradio as gr
import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emotional_memory import (
    EmotionalMemory,
    InMemoryStore,
    KeywordAppraisalEngine,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
)
from emotional_memory.integrations import (
    EmotionalMemoryChatHistory,
    recommended_conversation_policy,
)
from emotional_memory.llm_http import OpenAICompatibleLLMConfig, make_httpx_llm

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
# Appraisal engine — LLM if API key present, keyword fallback otherwise
# ---------------------------------------------------------------------------


def _make_appraisal_engine() -> tuple[KeywordAppraisalEngine | LLMAppraisalEngine, str]:
    config = OpenAICompatibleLLMConfig.from_env(os.environ)
    if config is not None:
        try:
            llm = make_httpx_llm(config)
        except ImportError:
            llm = None
        if llm is not None:
            engine = LLMAppraisalEngine(
                llm=llm,
                config=LLMAppraisalConfig(cache_size=64, fallback_on_error=True),
            )
            return engine, f"🧠 LLM appraisal active (`{config.model}`)"
    engine = KeywordAppraisalEngine()
    return (
        engine,
        "📝 Keyword fallback active "
        "(English only — set `EMOTIONAL_MEMORY_LLM_API_KEY` for multilingual)",
    )


# ---------------------------------------------------------------------------
# Global state (one instance per Gradio session via State)
# ---------------------------------------------------------------------------


def _build_engine() -> tuple[EmotionalMemory, EmotionalMemoryChatHistory, str]:
    appraisal_engine, mode = _make_appraisal_engine()
    em = EmotionalMemory(
        store=InMemoryStore(),
        embedder=_make_embedder(),
        appraisal_engine=appraisal_engine,
    )
    history = EmotionalMemoryChatHistory(em, message_policy=recommended_conversation_policy)
    return em, history, mode


# ---------------------------------------------------------------------------
# PAD plot
# ---------------------------------------------------------------------------

_PAD_HISTORY_MAX = 40


def _pad_plot(pad_history: list[tuple[float, float, float]]) -> Image.Image:
    """Return a PIL Image of the PAD trajectory plot."""
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
    return Image.open(buf)


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

MAX_MSG_PER_SESSION = 20


def _is_recall_request(text: str) -> bool:
    normalized = text.strip().lower()
    return (
        normalized == "recall"
        or normalized.startswith("recall ")
        or normalized == "remember"
        or normalized.startswith("remember ")
    )


def chat(
    user_msg: str,
    chat_history: list[dict[str, str]],
    em_state: EmotionalMemory,
    history_state: EmotionalMemoryChatHistory,
    pad_history: list[tuple[float, float, float]],
    msg_count: int,
) -> tuple[
    list[dict[str, str]],
    EmotionalMemory,
    EmotionalMemoryChatHistory,
    list[tuple[float, float, float]],
    Image.Image,
    str,
    int,
]:
    if not user_msg.strip():
        plot = _pad_plot(pad_history) if pad_history else _pad_plot([(0.0, 0.5, 0.0)])
        return chat_history, em_state, history_state, pad_history, plot, "", msg_count

    if msg_count >= MAX_MSG_PER_SESSION:
        gr.Info(
            f"Session limit of {MAX_MSG_PER_SESSION} messages reached. "
            "Click '🔄 New session' to continue."
        )
        return (
            chat_history,
            em_state,
            history_state,
            pad_history,
            _pad_plot(pad_history),
            "",
            msg_count,
        )

    # encode via history adapter — calls em.encode() internally
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
    if _is_recall_request(user_msg):
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
    return chat_history, em_state, history_state, pad_history, plot, "", msg_count + 1


def reset_session() -> tuple[
    list,
    EmotionalMemory,
    EmotionalMemoryChatHistory,
    list,
    Image.Image,
    str,
    int,
]:
    em, history, mode = _build_engine()
    plot = _pad_plot([(0.0, 0.5, 0.0)])
    return [], em, history, [(0.0, 0.5, 0.0)], plot, mode, 0


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_DESCRIPTION = """
# 🧠 Emotional Memory — Affective Field Theory Demo

Every message you send is encoded with a full **affective fingerprint** (valence, arousal,
appraisal, mood, resonance links). The PAD plot shows your conversation's emotional trajectory
in real time. Type **"recall X"** to trigger mood-congruent retrieval.

Built with [`emotional-memory`](https://github.com/gianlucamazza/emotional-memory) v0.6.0 · \
[PyPI](https://pypi.org/project/emotional-memory/) · \
[GitHub](https://github.com/gianlucamazza/emotional-memory) · \
[Zenodo DOI](https://doi.org/10.5281/zenodo.19636355)
"""

_EXAMPLES = [
    "I succeeded! My project is complete and it's a great personal victory.",
    "I failed and made a terrible mistake — everything is broken.",
    "There is danger here — this crisis is a serious risk I cannot handle.",
    "How surprising and unexpected! I am completely amazed by this news.",
    "recall happy moments",
]

with gr.Blocks(theme=gr.themes.Soft(), title="Emotional Memory Demo") as demo:
    gr.Markdown(_DESCRIPTION)

    em_state = gr.State()
    history_state = gr.State()
    pad_history = gr.State([])
    msg_count = gr.State(0)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation", height=420, type="messages", allow_tags=False
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Type a message… try expressing joy, fear, sadness, or calm.",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(examples=_EXAMPLES, inputs=msg_box)
            reset_btn = gr.Button("🔄 New session", variant="secondary", size="sm")
            appraisal_badge = gr.Markdown("", elem_id="appraisal-badge")

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
        outputs=[
            chatbot,
            em_state,
            history_state,
            pad_history,
            pad_plot,
            appraisal_badge,
            msg_count,
        ],
    )

    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, em_state, history_state, pad_history, msg_count],
        outputs=[chatbot, em_state, history_state, pad_history, pad_plot, msg_box, msg_count],
        concurrency_limit=2,
    )
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, em_state, history_state, pad_history, msg_count],
        outputs=[chatbot, em_state, history_state, pad_history, pad_plot, msg_box, msg_count],
        concurrency_limit=2,
    )
    reset_btn.click(
        fn=reset_session,
        outputs=[
            chatbot,
            em_state,
            history_state,
            pad_history,
            pad_plot,
            appraisal_badge,
            msg_count,
        ],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
