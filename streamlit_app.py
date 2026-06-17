"""
SOPAssist — Streamlit UI
GxP Regulatory Intelligence Platform

Runs the full RAG pipeline in-process (no HTTP server required).
Start with:  streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

import anthropic

from agent.qa_agent import SOPQAAgent
from config import settings
from ingestion.chroma_store import ChromaSOPStore
from ingestion.embedder import SOPEmbedder
from retrieval.confidence_scorer import ConfidenceScorer
from retrieval.retriever import VersionAwareRetriever

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SOPAssist — GxP Intelligence",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"] { background: #0f1d36; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; font-size: 0.78rem; }

.source-card {
    background: #f1f5f9;
    border-left: 3px solid #3b82f6;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.5rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    line-height: 1.5;
}
.citation-pill {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1px 8px;
    font-size: 0.76rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 3px;
}
.conf-meta { color: #94a3b8; font-size: 0.75rem; }
.badge-HIGH   { color: #059669; font-weight: 700; }
.badge-MEDIUM { color: #d97706; font-weight: 700; }
.badge-LOW    { color: #dc2626; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Pipeline loader (cached per session) ──────────────────────────────────────


@st.cache_resource(show_spinner="Loading SOPAssist pipeline…")
def _load_agent() -> SOPQAAgent:
    embedder = SOPEmbedder(settings.EMBEDDING_MODEL)
    store = ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    if not store.collection_exists_and_populated():
        st.error(
            "ChromaDB collection is empty. Run `python ingestion/run_ingestion.py` first.",
            icon="❌",
        )
        st.stop()
    retriever = VersionAwareRetriever(chroma_store=store, embedder=embedder)
    scorer = ConfidenceScorer(threshold=settings.CONFIDENCE_THRESHOLD)
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return SOPQAAgent(
        retriever=retriever,
        confidence_scorer=scorer,
        anthropic_client=client,
        model=settings.CLAUDE_MODEL,
    )


# ── Rendering helpers ─────────────────────────────────────────────────────────


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)} retrieved)", expanded=False):
        for src in sources:
            pct = int(src["similarity_score"] * 100)
            preview = src["text"][:300] + ("…" if len(src["text"]) > 300 else "")
            st.markdown(
                f"<div class='source-card'>"
                f"<span class='citation-pill'>{src['citation']}</span>&nbsp;"
                f"<strong>{src['section_title']}</strong>"
                f"<span style='color:#64748b;font-size:0.78rem'>"
                f" · {src['sop_title']} · {pct}% match</span>"
                f"<br/><span style='color:#334155'>{preview}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_conf_line(conf: dict | None, model: str = "") -> None:
    if not conf:
        return
    level = conf["level"]
    pct = int(conf["score"] * 100)
    st.markdown(
        f"<span class='conf-meta'>Confidence: "
        f"<span class='badge-{level}'>{level}</span> {pct}%"
        f"{' · ' + model if model else ''}</span>",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

DOMAIN_OPTIONS = {
    "All Domains": None,
    "GMP — Good Manufacturing Practice": "GMP",
    "GCP — Good Clinical Practice": "GCP",
    "GLP — Good Laboratory Practice": "GLP",
    "PV — Pharmacovigilance": "PV",
    "DI — Data Integrity": "DI",
}

with st.sidebar:
    st.markdown("## ⚕ SOPAssist")
    st.caption("GxP Regulatory Intelligence")
    st.divider()

    selected_label = st.selectbox("Domain Filter", list(DOMAIN_OPTIONS.keys()))
    domain_filter = DOMAIN_OPTIONS[selected_label]

    st.divider()

    if "last_conf" in st.session_state:
        conf = st.session_state.last_conf
        pct = int(conf["score"] * 100)
        level = conf["level"]
        bar_color = {"HIGH": "#10b981", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}.get(
            level, "#94a3b8"
        )
        st.markdown("### Confidence")
        st.markdown(
            f"<div style='margin-bottom:6px'>"
            f"<span class='badge-{level}'>{level}</span>"
            f"&nbsp;<span style='font-size:1.5rem;font-weight:700;color:#f1f5f9'>{pct}%</span>"
            f"</div>"
            f"<div style='background:#1e3a5f;border-radius:6px;height:10px;overflow:hidden'>"
            f"<div style='background:{bar_color};width:{pct}%;height:100%;border-radius:6px'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if conf.get("escalated"):
            st.warning(
                f"**Review Recommended**\n\n{conf.get('reason', '')}",
                icon="⚠️",
            )

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_conf", None)
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────

st.markdown("# SOPAssist")
st.caption(
    "Ask questions about your GxP Standard Operating Procedures. "
    "Answers are grounded in retrieved SOP sections with inline citations."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            _render_conf_line(msg.get("conf"), msg.get("model", ""))
            if msg.get("conf", {}).get("escalated"):
                st.warning(
                    f"**Quality Team Review Recommended**\n\n"
                    f"{msg['conf'].get('reason', '')}",
                    icon="⚠️",
                )

# Chat input
if user_input := st.chat_input("Ask about your SOPs…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching SOPs and generating answer…"):
            agent = _load_agent()
            resp = agent.answer(user_input, domain_filter=domain_filter)

        st.markdown(resp.answer)

        sources = [
            {
                "citation": r.citation,
                "section_title": r.section_title,
                "sop_title": r.sop_title,
                "similarity_score": r.similarity_score,
                "text": r.text,
            }
            for r in resp.retrieved_chunks
        ]
        _render_sources(sources)

        conf_dict = {
            "score": resp.confidence.score,
            "level": resp.confidence.level,
            "escalated": resp.confidence.should_escalate,
            "reason": resp.confidence.escalation_reason or "",
        }
        _render_conf_line(conf_dict, resp.model_used)

        if resp.confidence.should_escalate:
            st.warning(
                f"**Quality Team Review Recommended**\n\n"
                f"{resp.confidence.escalation_reason or ''}",
                icon="⚠️",
            )

    st.session_state.last_conf = conf_dict
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resp.answer,
            "sources": sources,
            "conf": conf_dict,
            "model": resp.model_used,
        }
    )
    st.rerun()
