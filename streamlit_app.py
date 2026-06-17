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
/* ── Base & Typography ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 15px;
}
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { display: none; }

/* ── App background ─────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f4f7f9; } /* Slightly cooler gray for lab tools */
.main .block-container {
    max-width: 880px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1423 0%, #16263f 100%);
    border-right: 1px solid #1e3150;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] hr { border-color: #1e3150 !important; opacity: 0.8; margin: 1.5rem 0; }
[data-testid="stSidebar"] label { 
    font-size: 0.7rem !important; 
    text-transform: uppercase;
    letter-spacing: 0.08em; 
    color: #64748b !important; 
    font-weight: 600; 
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    color: #f1f5f9 !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #fff !important;
    border-color: rgba(255,255,255,0.25) !important;
}
[data-testid="stSidebar"] .stWarning {
    background: rgba(217,119,6,0.1) !important;
    border: 1px solid rgba(217,119,6,0.3) !important;
    border-radius: 6px !important;
    font-size: 0.8rem;
    color: #fcd34d !important;
}

/* ── Chat messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04), 0 0 0 1px rgba(15, 23, 42, 0.03);
}
/* Style User Messages slightly differently */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    box-shadow: none;
}
[data-testid="stChatMessage"] p { line-height: 1.6; color: #1e293b; font-size: 0.95rem; }

/* ── Source cards ────────────────────────────────────────────────────────── */
.src-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.81rem;
    transition: box-shadow 0.2s ease;
}
.src-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    border-color: #cbd5e1;
}
.src-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.src-text { color: #475569; margin-top: 6px; font-size: 0.82rem; line-height: 1.5; }

/* ── Badges ──────────────────────────────────────────────────────────────── */
.cit-pill {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.7rem;
    font-weight: 600;
    white-space: nowrap;
}
.domain-tag {
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.match-score {
    color: #64748b;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.section-name { color: #0f172a; font-weight: 600; font-size: 0.8rem; }
.sop-name { color: #64748b; font-size: 0.75rem; font-weight: 500; }

/* ── Confidence line ─────────────────────────────────────────────────────── */
.conf-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #f1f5f9;
    flex-wrap: wrap;
}
.conf-label { color: #64748b; font-size: 0.75rem; font-weight: 500; }
.badge-HIGH   { color: #065f46; font-weight: 700; font-size: 0.7rem;
    background:#d1fae5; padding:2px 8px; border-radius:4px; border: 1px solid #a7f3d0; }
.badge-MEDIUM { color: #92400e; font-weight: 700; font-size: 0.7rem;
    background:#fef3c7; padding:2px 8px; border-radius:4px; border: 1px solid #fde68a; }
.badge-LOW    { color: #991b1b; font-weight: 700; font-size: 0.7rem;
    background:#fee2e2; padding:2px 8px; border-radius:4px; border: 1px solid #fecaca; }
.conf-pct { color: #334155; font-size: 0.75rem; font-weight: 700; }
.conf-model { color: #94a3b8; font-size: 0.7rem; font-family: monospace; }

/* ── Welcome screen ──────────────────────────────────────────────────────── */
.welcome-wrap { text-align: center; padding: 2rem 0 3rem; }
.welcome-icon { font-size: 2.5rem; margin-bottom: 1rem; color: #3b82f6; }
.welcome-title {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.welcome-sub { color: #64748b; font-size: 0.95rem; max-width: 540px; margin: 0 auto 2.5rem; line-height: 1.5; }
.ex-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 1rem; }
.ex-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.25rem;
    text-align: left;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    cursor: pointer;
    transition: all 0.2s ease;
}
.ex-card:hover { 
    border-color: #93c5fd; 
    box-shadow: 0 6px 16px rgba(59,130,246,0.08); 
    transform: translateY(-2px);
}
.ex-domain { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 8px; }
.ex-text { color: #334155; font-size: 0.85rem; line-height: 1.4; font-weight: 500; }

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    border-top: 1px solid #e2e8f0 !important;
    border-radius: 0 !important;
    margin-top: 1rem !important;
    padding-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: #64748b !important;
    font-weight: 600 !important;
    padding: 0 !important;
}
[data-testid="stExpander"] summary:hover { color: #3b82f6 !important; }
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

@st.cache_resource(show_spinner="Initializing SOPAssist Pipeline…")
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
    with st.expander(f"📄 View {len(sources)} Retrieved Sources", expanded=False):
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
                f"  <div style='margin-top:8px'>"
                f"    <span class='match-score'>▰▰▰ {pct}% semantic match</span>"
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
        f"<span class='conf-model'>{model}</span>" if model else ""
    )
    st.markdown(
        f"<div class='conf-row'>"
        f"  <span class='conf-label'>Retrieval Confidence</span>"
        f"  <span class='badge-{level}'>{level}</span>"
        f"  <span class='conf-pct'>{pct}%</span>"
        f"  {model_str}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _welcome_screen() -> str | None:
    st.markdown(
        "<div class='welcome-wrap'>"
        "<div class='welcome-icon'>⚕️</div>"
        "<div class='welcome-title'>SOPAssist</div>"
        "<div class='welcome-sub'>"
        "Ask procedural questions about your GxP Standard Operating Procedures. "
        "All answers are generated directly from verified SOP text with full citations."
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#64748b; margin-bottom: 0.5rem; text-transform:uppercase; letter-spacing:0.05em;'>Example Queries</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (domain, question) in enumerate(EXAMPLES):
        with cols[i % 2]:
            # We wrap the st.button in a styled div via markdown trickery, 
            # but Streamlit buttons don't accept raw HTML easily. 
            # We keep the standard button but let our CSS handle the card look.
            if st.button(
                f"[{domain}] {question}",
                key=f"ex_{i}",
                use_container_width=True,
            ):
                return question
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='padding:0.5rem 0 0.5rem'>"
        "<span style='font-size:1.4rem;font-weight:700;color:#f8fafc;letter-spacing:-0.02em'>"
        "⚕️ SOPAssist</span><br>"
        "<span style='font-size:0.7rem;color:#94a3b8;letter-spacing:0.05em;text-transform:uppercase'>"
        "GxP Intelligence Platform</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='display:flex;align-items:center;gap:6px;margin:0.2rem 0 1rem'>"
        "<span style='width:6px;height:6px;border-radius:50%;background:#10b981;"
        "display:inline-block;box-shadow:0 0 8px rgba(16,185,129,0.6)'></span>"
        "<span style='font-size:0.75rem;color:#94a3b8; font-weight:500'>System Online</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        "<span style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
        "color:#94a3b8;font-weight:600'>Scope Filter</span>",
        unsafe_allow_html=True,
    )
    selected_label = st.selectbox(
        "domain_select",
        list(DOMAIN_OPTIONS.keys()),
        label_visibility="collapsed",
    )
    domain_filter = DOMAIN_OPTIONS[selected_label]

    domain_desc = {
        None:  "Querying across all active SOPs",
        "GMP": "Good Manufacturing Practice",
        "GCP": "Good Clinical Practice",
        "GLP": "Good Laboratory Practice",
        "PV":  "Pharmacovigilance",
        "DI":  "Data Integrity",
    }
    st.markdown(
        f"<span style='font-size:0.75rem;color:#64748b'>{domain_desc[domain_filter]}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Confidence meter
    if "last_conf" in st.session_state:
        conf = st.session_state.last_conf
        pct = int(conf["score"] * 100)
        level = conf["level"]
        bar_color = {"HIGH": "#10b981", "MEDIUM": "#fbbf24", "LOW": "#ef4444"}.get(level, "#94a3b8")
        bg_color  = {"HIGH": "#064e3b", "MEDIUM": "#78350f", "LOW": "#7f1d1d"}.get(level, "#1e293b")
        txt_color = {"HIGH": "#34d399", "MEDIUM": "#fbbf24", "LOW": "#f87171"}.get(level, "#cbd5e1")

        st.markdown(
            "<span style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
            "color:#94a3b8;font-weight:600'>Last Retrieval Confidence</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin:12px 0 6px;display:flex;align-items:baseline;gap:8px'>"
            f"  <span style='font-size:2rem;font-weight:700;color:#f8fafc;line-height:1'>{pct}%</span>"
            f"  <span style='background:{bg_color};color:{txt_color};border-radius:4px;"
            f"    padding:2px 8px;font-size:0.65rem;font-weight:700; border: 1px solid {bar_color}40'>{level}</span>"
            f"</div>"
            f"<div style='background:rgba(0,0,0,0.3);border-radius:4px;height:4px;overflow:hidden;margin-bottom:8px;"
            f"box-shadow: inset 0 1px 2px rgba(0,0,0,0.2)'>"
            f"  <div style='background:{bar_color};width:{pct}%;height:100%;border-radius:4px;"
            f"    transition:width 0.5s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 8px {bar_color}80'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if conf.get("escalated"):
            st.warning(f"⚠️ Quality Review Flag\n\n{conf.get('reason', '')}")

    st.divider()

    if st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("last_conf", None)
        st.rerun()

    st.markdown(
        "<p style='font-size:0.7rem;color:#475569;line-height:1.5;margin-top:2rem'>"
        "SOPAssist enforces ALCOA+ data integrity standards. All outputs must be verified against primary records before GxP execution."
        "</p>",
        unsafe_allow_html=True,
    )

# ── Main chat area ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    clicked = _welcome_screen()
    if clicked:
        st.session_state.messages.append({"role": "user", "content": clicked})
        st.session_state["_pending"] = clicked
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            _render_conf_line(msg.get("conf"), msg.get("model", ""))
            if msg.get("conf", {}).get("escalated"):
                st.warning(
                    f"⚠️ Quality Team Review Recommended\n\n{msg['conf'].get('reason', '')}",
                )

question = st.session_state.pop("_pending", None)

if not question:
    typed = st.chat_input("Ask a procedural question...")
    if typed:
        question = typed
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

if question:
    with st.chat_message("assistant"):
        with st.spinner("Analyzing SOP corpus..."):
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
                f"⚠️ Quality Team Review Recommended\n\n{resp.confidence.escalation_reason or ''}",
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