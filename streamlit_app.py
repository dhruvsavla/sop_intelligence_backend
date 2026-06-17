"""
SOPAssist — Streamlit UI
GxP Regulatory Intelligence Platform

Run with:  streamlit run streamlit_app.py
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

st.markdown("""
<style>
/* ── Base ──────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 15px;
}
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { display: none; }

/* ── App background ─────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
.main .block-container {
    max-width: 860px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2f52 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] hr { border-color: #1e3a5f !important; opacity: 0.6; }
[data-testid="stSidebar"] label { font-size: 0.72rem !important; text-transform: uppercase;
    letter-spacing: 0.08em; color: #64748b !important; font-weight: 600; }
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.2) !important;
}
[data-testid="stSidebar"] .stWarning {
    background: rgba(217,119,6,0.15) !important;
    border: 1px solid rgba(217,119,6,0.3) !important;
    border-radius: 8px !important;
    font-size: 0.8rem;
}

/* ── Chat messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.04);
}
[data-testid="stChatMessage"] p { line-height: 1.7; color: #1e293b; }

/* ── Source cards ────────────────────────────────────────────────────────── */
.src-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.81rem;
    line-height: 1.65;
}
.src-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 5px;
}
.src-text { color: #475569; margin-top: 5px; font-size: 0.8rem; line-height: 1.6; }

/* ── Badges ──────────────────────────────────────────────────────────────── */
.cit-pill {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 0.7rem;
    font-weight: 700;
    white-space: nowrap;
    display: inline-block;
}
.domain-tag {
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.match-score {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 500;
}
.section-name { color: #1e293b; font-weight: 600; font-size: 0.82rem; }
.sop-name { color: #64748b; font-size: 0.75rem; }

/* ── Confidence line ─────────────────────────────────────────────────────── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #f1f5f9;
    flex-wrap: wrap;
}
.conf-label { color: #94a3b8; font-size: 0.73rem; }
.badge-HIGH   { color: #059669; font-weight: 700; font-size: 0.73rem;
    background:#d1fae5; padding:2px 7px; border-radius:20px; }
.badge-MEDIUM { color: #92400e; font-weight: 700; font-size: 0.73rem;
    background:#fef3c7; padding:2px 7px; border-radius:20px; }
.badge-LOW    { color: #991b1b; font-weight: 700; font-size: 0.73rem;
    background:#fee2e2; padding:2px 7px; border-radius:20px; }
.conf-pct { color: #475569; font-size: 0.73rem; font-weight: 600; }
.conf-model { color: #cbd5e1; font-size: 0.7rem; }

/* ── Welcome screen ──────────────────────────────────────────────────────── */
.welcome-wrap { text-align: center; padding: 1rem 0 2rem; }
.welcome-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.welcome-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.4rem;
}
.welcome-sub { color: #64748b; font-size: 0.9rem; max-width: 520px; margin: 0 auto 2rem; line-height: 1.6; }
.ex-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1rem; }
.ex-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    text-align: left;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    cursor: pointer;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.ex-card:hover { border-color: #93c5fd; box-shadow: 0 4px 12px rgba(59,130,246,0.1); }
.ex-domain { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 5px; }
.ex-text { color: #374151; font-size: 0.82rem; line-height: 1.5; }

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin-top: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: #475569 !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

DOMAIN_OPTIONS = {
    "All Domains": None,
    "GMP — Manufacturing Practice": "GMP",
    "GCP — Clinical Practice": "GCP",
    "GLP — Laboratory Practice": "GLP",
    "PV — Pharmacovigilance": "PV",
    "DI — Data Integrity": "DI",
}

DOMAIN_COLORS = {
    "GMP": ("#d1fae5", "#065f46"),
    "GCP": ("#dbeafe", "#1e3a8a"),
    "GLP": ("#fef3c7", "#78350f"),
    "PV":  ("#fce7f3", "#831843"),
    "DI":  ("#ede9fe", "#4c1d95"),
}

EXAMPLES = [
    ("GMP", "What are the cleaning validation requirements for manufacturing equipment?"),
    ("GCP", "What is the required timeframe for reporting a SUSAR to competent authorities?"),
    ("GLP", "How must errors be corrected in a GLP laboratory notebook?"),
    ("PV",  "What elements must be included in a Periodic Benefit-Risk Evaluation Report?"),
    ("DI",  "What controls are required for audit trail integrity in electronic systems?"),
    ("GMP", "What documentation is required before batch release for distribution?"),
]

# ── Pipeline (cached) ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising SOPAssist…")
def _load_agent() -> SOPQAAgent:
    embedder = SOPEmbedder(settings.EMBEDDING_MODEL)
    store = ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    if not store.collection_exists_and_populated():
        st.error("ChromaDB collection is empty. Run ingestion/run_ingestion.py first.")
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

def _domain_tag(domain: str) -> str:
    bg, fg = DOMAIN_COLORS.get(domain, ("#f1f5f9", "#475569"))
    return (
        f"<span class='domain-tag' style='background:{bg};color:{fg}'>{domain}</span>"
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 {len(sources)} source sections retrieved", expanded=False):
        for src in sources:
            pct = int(src["similarity_score"] * 100)
            preview = src["text"][:320] + ("…" if len(src["text"]) > 320 else "")
            domain = src.get("domain", "")
            st.markdown(
                f"<div class='src-card'>"
                f"  <div class='src-meta'>"
                f"    <span class='cit-pill'>{src['citation']}</span>"
                f"    {_domain_tag(domain) if domain else ''}"
                f"    <span class='section-name'>{src['section_title']}</span>"
                f"  </div>"
                f"  <div class='sop-name'>{src['sop_title']}</div>"
                f"  <div class='src-text'>{preview}</div>"
                f"  <div style='margin-top:6px'>"
                f"    <span class='match-score'>▲ {pct}% relevance</span>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_conf_line(conf: dict | None, model: str = "") -> None:
    if not conf:
        return
    level = conf["level"]
    pct = int(conf["score"] * 100)
    model_str = (
        f"<span class='conf-model'>· {model}</span>" if model else ""
    )
    st.markdown(
        f"<div class='conf-row'>"
        f"  <span class='conf-label'>Confidence</span>"
        f"  <span class='badge-{level}'>{level}</span>"
        f"  <span class='conf-pct'>{pct}%</span>"
        f"  {model_str}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _welcome_screen() -> str | None:
    """Render the welcome/empty-state screen. Returns a question if a card is clicked."""
    st.markdown(
        "<div class='welcome-wrap'>"
        "<div class='welcome-icon'>⚕</div>"
        "<div class='welcome-title'>SOPAssist</div>"
        "<div class='welcome-sub'>"
        "Ask any question about your GxP Standard Operating Procedures. "
        "Answers are grounded in verified SOP sections with precise citations."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Try an example question:**")
    cols = st.columns(2)
    for i, (domain, question) in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(
                f"[{domain}]  {question}",
                key=f"ex_{i}",
                use_container_width=True,
            ):
                return question
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='padding:0.5rem 0 0.25rem'>"
        "<span style='font-size:1.35rem;font-weight:700;color:#f1f5f9;letter-spacing:-0.01em'>"
        "⚕ SOPAssist</span><br>"
        "<span style='font-size:0.72rem;color:#64748b;letter-spacing:0.06em;text-transform:uppercase'>"
        "GxP Regulatory Intelligence</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='display:flex;align-items:center;gap:6px;margin:0.6rem 0 1rem'>"
        "<span style='width:7px;height:7px;border-radius:50%;background:#10b981;"
        "display:inline-block;box-shadow:0 0 0 2px rgba(16,185,129,0.25)'></span>"
        "<span style='font-size:0.72rem;color:#64748b'>Pipeline ready</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "<span style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
        "color:#475569;font-weight:600'>Domain Filter</span>",
        unsafe_allow_html=True,
    )
    selected_label = st.selectbox(
        "domain_select",
        list(DOMAIN_OPTIONS.keys()),
        label_visibility="collapsed",
    )
    domain_filter = DOMAIN_OPTIONS[selected_label]

    # Domain description blurb
    domain_desc = {
        None:  "Searching all GxP domains",
        "GMP": "Good Manufacturing Practice",
        "GCP": "Good Clinical Practice",
        "GLP": "Good Laboratory Practice",
        "PV":  "Pharmacovigilance",
        "DI":  "Data Integrity",
    }
    st.markdown(
        f"<span style='font-size:0.72rem;color:#475569'>{domain_desc[domain_filter]}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Confidence meter
    if "last_conf" in st.session_state:
        conf = st.session_state.last_conf
        pct = int(conf["score"] * 100)
        level = conf["level"]
        bar_color = {"HIGH": "#10b981", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}.get(level, "#94a3b8")
        bg_color  = {"HIGH": "#d1fae5", "MEDIUM": "#fef3c7", "LOW": "#fee2e2"}.get(level, "#f1f5f9")
        txt_color = {"HIGH": "#065f46", "MEDIUM": "#78350f", "LOW": "#7f1d1d"}.get(level, "#475569")

        st.markdown(
            "<span style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
            "color:#475569;font-weight:600'>Last Response</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin:8px 0 4px;display:flex;align-items:baseline;gap:8px'>"
            f"  <span style='font-size:1.9rem;font-weight:800;color:#f1f5f9;line-height:1'>{pct}%</span>"
            f"  <span style='background:{bg_color};color:{txt_color};border-radius:20px;"
            f"    padding:2px 9px;font-size:0.72rem;font-weight:700'>{level}</span>"
            f"</div>"
            f"<div style='background:rgba(255,255,255,0.08);border-radius:6px;height:6px;overflow:hidden;margin-bottom:4px'>"
            f"  <div style='background:{bar_color};width:{pct}%;height:100%;border-radius:6px;"
            f"    transition:width 0.4s ease'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if conf.get("escalated"):
            st.warning(f"⚠ Review Recommended\n\n{conf.get('reason', '')}")

    st.divider()

    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_conf", None)
        st.rerun()

    st.markdown(
        "<p style='font-size:0.68rem;color:#334155;line-height:1.6;margin-top:1.5rem'>"
        "Answers grounded in SOP citations only.<br>"
        "Always verify with the current approved document."
        "</p>",
        unsafe_allow_html=True,
    )

# ── Main chat area ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome screen — shown when conversation is empty
if not st.session_state.messages:
    clicked = _welcome_screen()
    if clicked:
        st.session_state.messages.append({"role": "user", "content": clicked})
        st.session_state["_pending"] = clicked
        st.rerun()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            _render_conf_line(msg.get("conf"), msg.get("model", ""))
            if msg.get("conf", {}).get("escalated"):
                st.warning(
                    f"⚠ Quality Team Review Recommended\n\n{msg['conf'].get('reason', '')}",
                )

# ── Answer generation ─────────────────────────────────────────────────────────
# Triggered either by a chat input or by clicking a welcome-screen example button.

question = st.session_state.pop("_pending", None)

if not question:
    typed = st.chat_input("Ask a question about your SOPs…")
    if typed:
        question = typed
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

if question:
    with st.chat_message("assistant"):
        with st.spinner("Searching SOPs…"):
            agent = _load_agent()
            resp = agent.answer(question, domain_filter=domain_filter)

        st.markdown(resp.answer)

        sources = [
            {
                "citation": r.citation,
                "section_title": r.section_title,
                "sop_title": r.sop_title,
                "domain": r.domain,
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
                f"⚠ Quality Team Review Recommended\n\n{resp.confidence.escalation_reason or ''}",
            )

    st.session_state.last_conf = conf_dict
    st.session_state.messages.append({
        "role": "assistant",
        "content": resp.answer,
        "sources": sources,
        "conf": conf_dict,
        "model": resp.model_used,
    })
    st.rerun()
