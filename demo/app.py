"""Emotional Memory — Affective Field Theory live demo.

Runs as a Hugging Face Space (sdk: gradio) or locally::

    make install-demo
    make demo-run

Set EMOTIONAL_MEMORY_LLM_API_KEY (+ optionally EMOTIONAL_MEMORY_LLM_MODEL and
EMOTIONAL_MEMORY_LLM_BASE_URL) to enable LLM-backed appraisal (multilingual,
full AFT pipeline).  Without it the demo falls back to KeywordAppraisalEngine
(rule-based, English only).

This module reads LLM configuration from the process environment only. It does
not call ``load_dotenv()`` itself; use ``make demo-run`` or export variables in
your shell before running ``uv run python demo/app.py`` for local `.env`
convenience. ``demo/requirements.txt`` is reserved for the deployed Space
runtime overlay.
"""

from __future__ import annotations

import asyncio.base_events
import hashlib
import importlib.util
import io
import logging
import os

# No telemetry from this process. Gradio / hub clients honour these flags.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
from functools import lru_cache

logger = logging.getLogger(__name__)


def _patch_event_loop_cleanup() -> None:
    """Suppress a known Gradio/Python 3.11 cleanup traceback on interpreter shutdown.

    Gradio's async server leaves the event loop in a state where the default
    ``BaseEventLoop.__del__`` raises ``ValueError: Invalid file descriptor``
    during interpreter teardown.  This patch silently swallows only that specific
    error; all other exceptions are still propagated.  Applied once at import time
    via a sentinel attribute so re-imports are no-ops.
    """
    current_del = asyncio.base_events.BaseEventLoop.__del__
    if getattr(current_del, "__emotional_memory_patched__", False):
        return

    def _safe_del(self: object) -> None:
        try:
            current_del(self)
        except ValueError as exc:
            if "Invalid file descriptor" not in str(exc):
                raise

    _safe_del.__emotional_memory_patched__ = True
    asyncio.base_events.BaseEventLoop.__del__ = _safe_del  # type: ignore[method-assign]


_patch_event_loop_cleanup()

import gradio as gr
import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emotional_memory import (
    AppraisalEngine,
    EmotionalMemory,
    InMemoryStore,
    KeywordAppraisalEngine,
    LLMAppraisalConfig,
    LLMAppraisalEngine,
    __version__,
)
from emotional_memory.engine_shared import (
    DEFAULT_QUERY_AFFECT_GATE_TAU,
    is_query_affect_neutral,
    semantic_only_weights,
)
from emotional_memory.llm_http import (
    OpenAICompatibleLLMConfig,
    make_httpx_llm,
)

RETRIEVAL_MODE_MOOD_STATE = "Mood-state recall (mechanism demo)"
RETRIEVAL_MODE_QUERY_GATED = "Query-gated recall"

# ---------------------------------------------------------------------------
# Embedder — use SentenceTransformer if available, fall back to Hash
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _make_embedder():  # type: ignore[no-untyped-def]
    try:
        from emotional_memory.embedders import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder()
    except ImportError:
        logger.info("sentence_transformers not available — using hash embedder fallback")
        return _HashEmbedder()


def _embedder_mode_badge() -> str:
    if importlib.util.find_spec("sentence_transformers") is None:
        return (
            "🔎 Hash fallback retrieval active "
            "(approximate semantic recall — install `sentence-transformers` for stronger results)"
        )
    return "🔎 SentenceTransformer retrieval active (`all-MiniLM-L6-v2`)"


class _HashEmbedder:
    def embed(self, text: str) -> list[float]:
        d = hashlib.sha256(text.encode()).digest()
        return [(b / 127.5) - 1.0 for b in d[:64]]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# Appraisal engine — LLM if API key present, keyword fallback otherwise
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
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
        logger.info("httpx not available — falling back to keyword appraisal engine")
    engine = KeywordAppraisalEngine()
    return (
        engine,
        "📝 Keyword fallback active "
        "(English only — set `EMOTIONAL_MEMORY_LLM_API_KEY` for multilingual)",
    )


def _runtime_mode_badge() -> str:
    _, appraisal_mode = _make_appraisal_engine()
    return f"{appraisal_mode}  \n{_embedder_mode_badge()}"


# ---------------------------------------------------------------------------
# Global state (one isolated engine per retrieval mode, per Gradio session)
# ---------------------------------------------------------------------------


class DemoSession:
    """Per-session pair of ``EmotionalMemory`` engines, one per retrieval mode.

    Before any recall, non-recall turns encode once on the mood engine and
    replicate the full store and affective state into the gated engine. After
    a recall has diverged the modes, each engine encodes the new turn with the
    shared appraisal so reverse links and per-engine tags stay local. Recall
    write-backs and the ``observe(reply)`` that follows a recall stay on the
    mode that ran the recall, so a comparison is not confounded by the other
    mode's retrieval side-effects.
    """

    __slots__ = (
        "appraisal_engine",
        "appraisal_mode",
        "diverged",
        "embedder",
        "gated",
        "mood",
    )

    def __init__(
        self,
        mood: EmotionalMemory,
        gated: EmotionalMemory,
        embedder: object,
        appraisal_engine: AppraisalEngine,
        appraisal_mode: str = "",
    ) -> None:
        self.mood = mood
        self.gated = gated
        self.embedder = embedder
        self.appraisal_engine = appraisal_engine
        self.appraisal_mode = appraisal_mode
        self.diverged = False
        # Fresh engines have distinct timestamps; align them so the first
        # non-recall encode can replicate instead of encoding twice.
        _replicate_from(self.mood, self.gated)

    def __repr__(self) -> str:
        return (
            f"DemoSession(mood={self.mood!r}, gated={self.gated!r}, "
            f"appraisal_mode={self.appraisal_mode!r})"
        )

    def engine_for(self, retrieval_mode: str) -> EmotionalMemory:
        if retrieval_mode == RETRIEVAL_MODE_QUERY_GATED:
            return self.gated
        return self.mood

    def list_all(self) -> list[object]:
        return self.mood.list_all()

    def get_state(self) -> object:
        return self.mood.get_state()


def _new_engine(
    embedder: object,
    appraisal_engine: AppraisalEngine,
) -> EmotionalMemory:
    return EmotionalMemory(
        store=InMemoryStore(),
        embedder=embedder,
        appraisal_engine=appraisal_engine,
    )


def _build_engine() -> tuple[DemoSession, str]:
    appraisal_engine, appraisal_mode = _make_appraisal_engine()
    embedder = _make_embedder()
    session = DemoSession(
        mood=_new_engine(embedder, appraisal_engine),
        gated=_new_engine(embedder, appraisal_engine),
        embedder=embedder,
        appraisal_engine=appraisal_engine,
        appraisal_mode=appraisal_mode,
    )
    return session, f"{appraisal_mode}  \n{_embedder_mode_badge()}"


def _ensure_session(em_state: object) -> DemoSession:
    """Return a live session, rebuilding when Gradio handed us the factory."""
    if isinstance(em_state, DemoSession):
        return em_state
    session, _mode = _build_engine()
    return session


def _replicate_from(source: EmotionalMemory, dest: EmotionalMemory) -> None:
    dest.import_memories(source.export_memories(), overwrite=True)
    dest.load_state(source.save_state())


def _encode_user_both(session: DemoSession, content: str, metadata: dict[str, str]) -> None:
    """Appraise/encode once, then apply the same appraisal on the other mode.

    Before any recall, copy the full store and affective state with the public
    export/import + save/load API so ids stay aligned while the stores match.
    After a recall has diverged the modes, encode on the gated engine with the
    mood engine's appraisal: that writes reverse back-links on gated's existing
    memories and builds the new tag from gated's own state. A missing appraisal
    falls back to importing only the new memory, without a second appraisal.
    """
    memory = session.mood.encode(content, metadata=metadata)
    if not session.diverged:
        _replicate_from(session.mood, session.gated)
        return
    appraisal = memory.tag.appraisal
    if appraisal is None:
        logger.warning(
            "post-divergence encode has no stored appraisal; importing the new "
            "memory without a second appraisal"
        )
        dest_ids = {item["id"] for item in session.gated.export_memories()}
        new_items = [item for item in session.mood.export_memories() if item["id"] not in dest_ids]
        session.gated.import_memories(new_items)
        return
    session.gated.encode(content, appraisal=appraisal, metadata=metadata)


def _observe_both(session: DemoSession, content: str, metadata: dict[str, str]) -> None:
    tag = session.mood.observe(content, metadata=metadata)
    if not session.diverged:
        session.gated.load_state(session.mood.save_state())
        return
    session.gated.observe(content, appraisal=tag.appraisal, metadata=metadata)


def _scratch_from(session: DemoSession, source: EmotionalMemory) -> EmotionalMemory:
    scratch = _new_engine(session.embedder, session.appraisal_engine)
    _replicate_from(source, scratch)
    return scratch


def _memory_excerpt(memory: object) -> str:
    content = getattr(memory, "content", "")
    return f'"{content[:60]}…"'


def _clear_compare_output() -> str:
    return ""


def _env_flag(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_enable_ssr() -> bool:
    """Ignore platform defaults; allow SSR only via project-owned opt-in."""
    return _env_flag("EMOTIONAL_MEMORY_DEMO_SSR") is True


def _launch_kwargs() -> dict[str, object]:
    return {
        "show_error": True,
        "theme": _DEMO_THEME,
        "ssr_mode": _should_enable_ssr(),
    }


# ---------------------------------------------------------------------------
# PAD plot
# ---------------------------------------------------------------------------

_PAD_HISTORY_MAX = 40
_INITIAL_PAD_HISTORY: list[tuple[float, float, float]] = [(0.0, 0.5, 0.0)]
_INITIAL_CHAT_HISTORY: list[dict[str, str]] = []
_INITIAL_MSG_COUNT = 0


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


@lru_cache(maxsize=1)
def _initial_pad_plot() -> Image.Image:
    return _pad_plot(_INITIAL_PAD_HISTORY)


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

MAX_MSG_PER_SESSION = 20


def _parse_recall_query(text: str) -> str | None:
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return None
    if parts[0].lower() not in {"recall", "remember"}:
        return None
    return parts[1].strip() if len(parts) > 1 else ""


def _is_recall_request(text: str) -> bool:
    return _parse_recall_query(text) is not None


def _keyword_fallback_active(engine: object, mode: str) -> bool:
    return isinstance(engine, KeywordAppraisalEngine) or "Keyword fallback" in mode


def _retrieve_query_gated(
    em_state: EmotionalMemory,
    query: str,
    *,
    top_k: int = 3,
    appraisal_engine: AppraisalEngine | None = None,
    appraisal_mode: str | None = None,
) -> tuple[list[object], str, bool]:
    """Reproduce 0.18.1 ``retrieve_query_gated`` from one appraisal + public retrieve.

    Package 0.18.1 does not expose the gate decision (no return value, hook, or
    public appraise-query helper). Appraise once, then call the same public
    ``retrieve`` arms the library uses.
    """
    if appraisal_engine is None or appraisal_mode is None:
        factory_engine, factory_mode = _make_appraisal_engine()
        if appraisal_engine is None:
            appraisal_engine = factory_engine
        if appraisal_mode is None:
            appraisal_mode = factory_mode
    query_affect = appraisal_engine.appraise(query).to_core_affect()
    if is_query_affect_neutral(query_affect.valence, DEFAULT_QUERY_AFFECT_GATE_TAU):
        results = em_state.retrieve(
            query, top_k=top_k, precomputed_weights=semantic_only_weights()
        )
        branch = "neutral query, cosine only"
    else:
        results = em_state.retrieve(query, top_k=top_k, query_affect=query_affect)
        branch = "affect-aware"
    return results, branch, _keyword_fallback_active(appraisal_engine, appraisal_mode)


def _format_top_memories(results: list[object]) -> str:
    if not results:
        return "_No memories yet._"
    return "\n".join(
        f"{index}. {_memory_excerpt(memory)}" for index, memory in enumerate(results[:3], start=1)
    )


def compare_both_modes(
    query: str,
    em_state: object,
    retrieval_mode: str = RETRIEVAL_MODE_MOOD_STATE,
) -> str:
    """Diagnose one selected snapshot; retrieval writes only to disposable copies."""
    session = _ensure_session(em_state)
    raw = query.strip()
    if not raw:
        return "Type a query (or `recall …`) then click **Compare both modes**."
    parsed = _parse_recall_query(raw)
    retrieval_query = parsed if parsed else raw
    source = session.engine_for(retrieval_mode)
    mood_scratch = _scratch_from(session, source)
    gated_scratch = _scratch_from(session, source)
    mood_results = mood_scratch.retrieve(retrieval_query, top_k=3)
    try:
        gated_results, branch, keyword_fallback = _retrieve_query_gated(
            gated_scratch,
            retrieval_query,
            top_k=3,
            appraisal_engine=session.appraisal_engine,
            appraisal_mode=session.appraisal_mode,
        )
    except Exception as exc:
        logger.warning("compare-both gated path failed (%s)", exc)
        gated_results, branch, keyword_fallback = [], "unavailable", False
    fallback = (
        "\n\n_Query-gated comparison used the keyword appraisal fallback (English only)._"
        if keyword_fallback
        else ""
    )
    return (
        "### Compare both modes (snapshot — no live memory write-back)\n\n"
        f"Source snapshot: **{retrieval_mode}**\n\n"
        "Both strategies start from this selected engine's current memory and affective state. "
        "This is a point-in-time diagnostic, not a longitudinal benchmark.\n\n"
        f"Query: `{retrieval_query}`\n\n"
        f"**{RETRIEVAL_MODE_MOOD_STATE}**\n\n"
        f"{_format_top_memories(mood_results)}\n\n"
        f"**{RETRIEVAL_MODE_QUERY_GATED}** — {branch}\n\n"
        f"{_format_top_memories(gated_results)}"
        f"{fallback}"
    )


def chat(
    user_msg: str,
    chat_history: list[dict[str, str]],
    em_state: object,
    pad_history: list[tuple[float, float, float]],
    msg_count: int,
    retrieval_mode: str = RETRIEVAL_MODE_MOOD_STATE,
) -> tuple[
    list[dict[str, str]],
    DemoSession,
    list[tuple[float, float, float]],
    Image.Image,
    str,
    int,
]:
    session = _ensure_session(em_state)
    if not user_msg.strip():
        plot = _pad_plot(pad_history) if pad_history else _pad_plot([(0.0, 0.5, 0.0)])
        return chat_history, session, pad_history, plot, "", msg_count

    if msg_count >= MAX_MSG_PER_SESSION:
        gr.Info(
            f"Session limit of {MAX_MSG_PER_SESSION} messages reached. "
            "Click '🔄 New session' to continue."
        )
        return (
            chat_history,
            session,
            pad_history,
            _pad_plot(pad_history),
            "",
            msg_count,
        )

    recall_query = _parse_recall_query(user_msg)
    is_recall_request = recall_query is not None
    if not is_recall_request:
        try:
            _encode_user_both(session, user_msg, {"role": "user"})
        except Exception as exc:
            logger.exception("encode failed: %s", exc)
            error_reply = f"⚠️ Could not encode memory: {exc}"
            chat_history = [
                *chat_history,
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": error_reply},
            ]
            return chat_history, session, pad_history, _pad_plot(pad_history), "", msg_count + 1

    use_gated_recall = bool(
        is_recall_request and retrieval_mode == RETRIEVAL_MODE_QUERY_GATED and recall_query
    )
    if use_gated_recall:
        active = session.gated
    elif is_recall_request:
        active = session.mood
    else:
        active = session.engine_for(retrieval_mode)

    state = active.get_state()
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
        f"I have {len(active.list_all())} memories stored. "
        "Ask me to 'recall project success' to see semantic + mood-aware retrieval."
    )

    # Handle recall requests
    if is_recall_request:
        retrieval_query = recall_query if recall_query else user_msg
        gated_note = ""
        if use_gated_recall:
            try:
                appraisal_engine, appraisal_mode = _make_appraisal_engine()
                results, branch, keyword_fallback = _retrieve_query_gated(
                    session.gated,
                    retrieval_query,
                    top_k=3,
                    appraisal_engine=appraisal_engine,
                    appraisal_mode=appraisal_mode,
                )
                if keyword_fallback:
                    gated_note = (
                        " Query-gated recall is using the keyword appraisal fallback "
                        "(English only — set `EMOTIONAL_MEMORY_LLM_API_KEY` for LLM appraisal)."
                    )
                if results:
                    excerpts = "; ".join(_memory_excerpt(m) for m in results[:3])
                    reply = (
                        f"[{tone}] Top memories (query-gated, {branch}): {excerpts}{gated_note}"
                    )
                else:
                    reply = (
                        f"[{tone}] No memories yet — keep chatting to build them up!{gated_note}"
                    )
            except Exception as exc:
                logger.warning("query-gated recall failed (%s); using mood-state recall", exc)
                active = session.mood
                results = session.mood.retrieve(retrieval_query, top_k=3)
                if results:
                    excerpts = "; ".join(_memory_excerpt(m) for m in results[:3])
                    reply = (
                        f"[{tone}] Top memories matching your query and current mood: {excerpts} "
                        "(query-gated recall unavailable, used mood-state recall)"
                    )
                else:
                    reply = f"[{tone}] No memories yet — keep chatting to build them up!"
        else:
            results = session.mood.retrieve(retrieval_query, top_k=3)
            if results:
                excerpts = "; ".join(_memory_excerpt(m) for m in results[:3])
                if recall_query:
                    reply = (
                        f"[{tone}] Top memories matching your query and current mood: {excerpts}"
                    )
                elif retrieval_mode == RETRIEVAL_MODE_QUERY_GATED:
                    reply = (
                        f"[{tone}] Top memories matching your current mood "
                        f"(bare recall stays on mood-state recall): {excerpts}"
                    )
                else:
                    reply = f"[{tone}] Top memories matching your current mood: {excerpts}"
            else:
                reply = f"[{tone}] No memories yet — keep chatting to build them up!"

    try:
        if is_recall_request:
            # Write-backs already applied by retrieve(); observe the reply on
            # the same mode only so the other mode's state is untouched.
            active.observe(reply, metadata={"role": "assistant"})
            session.diverged = True
        else:
            _observe_both(session, reply, {"role": "assistant"})
    except Exception as exc:
        logger.warning("observe failed (non-fatal): %s", exc)

    state = active.get_state()
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
    return chat_history, session, pad_history, plot, "", msg_count + 1


def reset_session(
    *_args: object,
) -> tuple[
    list,
    DemoSession,
    list,
    Image.Image,
    str,
    int,
]:
    # Guard: API clients may pass the unbound State factory; ignore and rebuild.
    session, mode = _build_engine()
    plot = _initial_pad_plot().copy()
    return [], session, list(_INITIAL_PAD_HISTORY), plot, mode, 0


def _initial_em_state() -> DemoSession:
    session, _mode = _build_engine()
    return session


def _hide_state_factory_from_api(demo: gr.Blocks) -> None:
    """Keep the State factory load event off the public API listing.

    ``gr.State(value=_initial_em_state)`` is still a per-session factory: Gradio
    attaches a load event named after the callable. That event stays registered
    for session init; only ``api_visibility`` changes so ``/_initial_em_state``
    is not advertised in ``/gradio_api/info``.
    """
    for block_fn in demo.fns.values():
        if block_fn.fn is _initial_em_state:
            block_fn.api_visibility = "private"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

# Release metadata — managed by scripts/sync_release_metadata.py
# Edit release.toml, then run: make sync-metadata
_ZENODO_CONCEPT_DOI = "10.5281/zenodo.19972258"  # [ssot:concept_doi]
_REPO_URL = "https://github.com/gianlucamazza/emotional-memory"  # [ssot:repo_url]

_DESCRIPTION = f"""
# 🧠 Emotional Memory — Affective Field Theory Demo

Every message you send is encoded with a full **affective fingerprint** (valence, arousal,
appraisal, mood, resonance links). The PAD plot shows your conversation's emotional trajectory
in real time. Type **"recall project success"** for semantic + mood-aware retrieval,
or plain **"recall"** for a general mood-congruent pass.

Built with
[`emotional-memory`]({_REPO_URL}) v{__version__} · \
[PyPI](https://pypi.org/project/emotional-memory/) · \
[GitHub]({_REPO_URL}) · \
[Zenodo DOI](https://doi.org/{_ZENODO_CONCEPT_DOI})
"""

_EXAMPLES = [
    "I succeeded! My project is complete and it's a great personal victory.",
    "I failed and made a terrible mistake — everything is broken.",
    "There is danger here — this crisis is a serious risk I cannot handle.",
    "How surprising and unexpected! I am completely amazed by this news.",
    "recall project success",
]

_DEMO_THEME = gr.themes.Soft()


def build_demo() -> gr.Blocks:
    initial_runtime_badge = _runtime_mode_badge()

    with gr.Blocks(title="Emotional Memory Demo") as demo:
        gr.Markdown(_DESCRIPTION)

        em_state = gr.State(value=_initial_em_state)
        pad_history = gr.State(value=_INITIAL_PAD_HISTORY)
        msg_count = gr.State(value=_INITIAL_MSG_COUNT)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    # No type= kwarg: Gradio 6 removed it (messages is the only
                    # format); passing it raises TypeError at startup on the Space.
                    value=_INITIAL_CHAT_HISTORY,
                    label="Conversation",
                    height=420,
                    allow_tags=False,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type a message… try expressing joy, fear, sadness, or calm.",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                retrieval_mode = gr.Radio(
                    choices=[RETRIEVAL_MODE_MOOD_STATE, RETRIEVAL_MODE_QUERY_GATED],
                    value=RETRIEVAL_MODE_MOOD_STATE,
                    label="Retrieval mode",
                    info=(
                        "Mood-state recall scores against the running conversation affect. "
                        "Query-gated recall appraises the recall query and uses cosine-only "
                        "ranking when that query is affectively neutral."
                    ),
                )
                gr.Markdown(
                    "Normal recall updates memory (retrieval count, reconsolidation, "
                    "link strengthening) as part of the mechanism; each mode keeps its "
                    "own memory. Compare runs both strategies on disposable copies of the "
                    "selected mode's snapshot; it does not compare their accumulated histories."
                )
                compare_btn = gr.Button("Compare both modes", variant="secondary", size="sm")
                compare_output = gr.Markdown("")
                gr.Examples(examples=_EXAMPLES, inputs=msg_box)
                reset_btn = gr.Button("🔄 New session", variant="secondary", size="sm")
                appraisal_badge = gr.Markdown(initial_runtime_badge, elem_id="appraisal-badge")

            with gr.Column(scale=2):
                pad_plot = gr.Image(
                    value=_initial_pad_plot(),
                    label="PAD state trajectory",
                    type="pil",
                )
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
                pad_history,
                pad_plot,
                appraisal_badge,
                msg_count,
            ],
            api_name="init_session",
            api_visibility="private",
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_box, chatbot, em_state, pad_history, msg_count, retrieval_mode],
            outputs=[chatbot, em_state, pad_history, pad_plot, msg_box, msg_count],
            concurrency_limit=2,
            api_name="chat",
        )
        msg_box.submit(
            fn=chat,
            inputs=[msg_box, chatbot, em_state, pad_history, msg_count, retrieval_mode],
            outputs=[chatbot, em_state, pad_history, pad_plot, msg_box, msg_count],
            concurrency_limit=2,
            api_name="chat_submit",
            api_visibility="private",
        )
        compare_btn.click(
            fn=compare_both_modes,
            inputs=[msg_box, em_state, retrieval_mode],
            outputs=[compare_output],
            api_name="compare_both",
        )
        reset_btn.click(
            fn=reset_session,
            outputs=[
                chatbot,
                em_state,
                pad_history,
                pad_plot,
                appraisal_badge,
                msg_count,
            ],
            api_name="reset_session",
        )
        reset_btn.click(
            fn=_clear_compare_output,
            outputs=[compare_output],
            api_name="clear_compare",
            api_visibility="private",
        )

    _hide_state_factory_from_api(demo)
    return demo


if __name__ == "__main__":
    build_demo().launch(**_launch_kwargs())
