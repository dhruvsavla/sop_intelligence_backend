"""
SOPAssist — Evaluation Dashboard
Runs the 60-question benchmark against the live pipeline and displays results.
"""
import json
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.qa_agent import SOPQAAgent
from config import settings
from evaluation.eval_dataset import EVALUATION_DATASET
from ingestion.chroma_store import ChromaSOPStore
from ingestion.embedder import SOPEmbedder
from retrieval.confidence_scorer import ConfidenceScorer
from retrieval.retriever import VersionAwareRetriever

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SOPAssist — Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (matches main app) ────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
#MainMenu, footer, header { visibility: hidden; height: 0; }
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1a2f52 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.main .block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }

.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.04);
    text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
.metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
    color: #64748b; font-weight: 600; margin-top: 4px; }
.metric-target { font-size: 0.68rem; color: #94a3b8; margin-top: 2px; }

.pass-banner {
    background: #d1fae5; border: 1px solid #6ee7b7; border-radius: 12px;
    padding: 1rem 1.4rem; color: #065f46; font-weight: 600; font-size: 0.9rem;
}
.fail-banner {
    background: #fee2e2; border: 1px solid #fca5a5; border-radius: 12px;
    padding: 1rem 1.4rem; color: #7f1d1d; font-weight: 600; font-size: 0.9rem;
}
.q-row-pass { background: #f0fdf4 !important; }
.q-row-fail { background: #fff7ed !important; }
</style>
""", unsafe_allow_html=True)

# ── Pipeline (shared cache with main app) ─────────────────────────────────────

@st.cache_resource(show_spinner="Loading pipeline…")
def _load_agent() -> SOPQAAgent:
    embedder = SOPEmbedder(settings.EMBEDDING_MODEL)
    store = ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    retriever = VersionAwareRetriever(chroma_store=store, embedder=embedder)
    scorer = ConfidenceScorer(threshold=settings.CONFIDENCE_THRESHOLD)
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return SOPQAAgent(
        retriever=retriever,
        confidence_scorer=scorer,
        anthropic_client=client,
        model=settings.CLAUDE_MODEL,
    )

# ── Evaluation runner (sync, with live progress) ──────────────────────────────

CITATION_PATTERN = re.compile(r"\[SOP-[A-Z]+-\d+\s+v[\d.]+,\s+Section\s+[\d.]+\]")


def _run_evaluation(agent: SOPQAAgent, progress_bar, status_text, log_container) -> dict:
    total = len(EVALUATION_DATASET)
    rows = []
    citation_correct = 0
    escalation_correct_oos = 0
    false_escalations = 0
    confidence_scores = []
    times = []
    domain_results: dict[str, list[bool]] = {}
    log_lines: list[str] = []

    for i, q in enumerate(EVALUATION_DATASET):
        progress_bar.progress((i + 1) / total)
        status_text.markdown(
            f"<span style='font-size:0.82rem;color:#475569'>"
            f"Question {i+1} / {total} &nbsp;·&nbsp; {q['question'][:80]}…</span>",
            unsafe_allow_html=True,
        )

        try:
            resp = agent.answer(q["question"], domain_filter=q.get("domain"))
        except Exception as exc:
            rows.append({
                "ID": q["question_id"], "Domain": q.get("domain", ""), "Q": q["question"],
                "Answerable": q["is_answerable"], "Citation ✓": False,
                "Escalated": False, "Confidence": 0.0, "Level": "—",
                "Citations Found": "", "Time (ms)": 0, "Error": str(exc),
            })
            continue

        # Citation check
        expected = q.get("expected_citation_pattern")
        cit_ok = False
        if expected:
            for c in resp.citations:
                if re.search(expected, c):
                    cit_ok = True
                    break
            if not cit_ok and re.search(expected, resp.answer):
                cit_ok = True

        is_answerable = q["is_answerable"]
        if is_answerable and cit_ok:
            citation_correct += 1
        if not is_answerable and resp.escalated:
            escalation_correct_oos += 1
        if is_answerable and resp.escalated and not resp.citations:
            false_escalations += 1

        confidence_scores.append(resp.confidence.score)
        times.append(resp.processing_time_ms)

        domain = q.get("domain", "")
        if domain:
            domain_results.setdefault(domain, []).append(cit_ok)

        icon = "✅" if cit_ok else "❌"
        log_lines.append(
            f"{icon} **{q['question_id']}** ({domain}) — "
            f"Confidence: {resp.confidence.level} {int(resp.confidence.score*100)}% "
            f"| Escalated: {resp.escalated}"
        )
        log_container.markdown("\n\n".join(log_lines[-15:]))  # rolling last 15

        rows.append({
            "ID": q["question_id"],
            "Domain": domain,
            "Question": q["question"],
            "Answerable": "Yes" if is_answerable else "No (OOS)",
            "Citation ✓": "✅" if cit_ok else "❌",
            "Escalated": "Yes" if resp.escalated else "No",
            "Confidence": round(resp.confidence.score, 3),
            "Level": resp.confidence.level,
            "Citations Found": ", ".join(resp.citations),
            "Time (ms)": resp.processing_time_ms,
        })

        time.sleep(0.3)  # avoid rate limits

    n_answerable = sum(1 for q in EVALUATION_DATASET if q["is_answerable"])
    n_oos = sum(1 for q in EVALUATION_DATASET if not q["is_answerable"])

    domain_breakdown = {
        d: round(sum(v) / len(v) * 100, 1) for d, v in domain_results.items() if v
    }

    report = {
        "total": total,
        "citation_accuracy": round(citation_correct / n_answerable * 100, 1) if n_answerable else 0,
        "escalation_accuracy": round(escalation_correct_oos / n_oos * 100, 1) if n_oos else 0,
        "false_escalation_rate": round(false_escalations / n_answerable * 100, 1) if n_answerable else 0,
        "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 3) if confidence_scores else 0,
        "avg_time_ms": round(sum(times) / len(times), 0) if times else 0,
        "domain_breakdown": domain_breakdown,
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "passed": citation_correct / n_answerable >= 0.90 and escalation_correct_oos / n_oos >= 0.80
        if n_answerable and n_oos else False,
    }

    # Save JSON report
    report_path = Path(__file__).parent.parent / "reports" / "evaluation_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    return report


# ── Report display ────────────────────────────────────────────────────────────

def _show_report(report: dict) -> None:
    # Pass / Fail banner
    if report.get("passed"):
        st.markdown(
            "<div class='pass-banner'>✅ &nbsp; PASS — Meets production readiness criteria "
            "(Citation ≥ 90% &amp; Escalation ≥ 80%)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='fail-banner'>⚠ &nbsp; NEEDS IMPROVEMENT — "
            "Review citation accuracy or escalation handling</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<p style='color:#94a3b8;font-size:0.75rem;margin-top:6px'>"
                f"Run at {report.get('timestamp','')} · {report['total']} questions</p>",
                unsafe_allow_html=True)

    st.markdown("---")

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        (col1, f"{report['citation_accuracy']}%",  "Citation Accuracy",   "Target ≥ 90%",
         "#059669" if report["citation_accuracy"] >= 90 else "#dc2626"),
        (col2, f"{report['escalation_accuracy']}%", "Escalation Accuracy", "Target ≥ 80%",
         "#059669" if report["escalation_accuracy"] >= 80 else "#dc2626"),
        (col3, f"{report['false_escalation_rate']}%", "False Escalations",  "Lower is better",
         "#059669" if report["false_escalation_rate"] < 15 else "#dc2626"),
        (col4, f"{report['avg_confidence']:.2f}",  "Avg Confidence",      "0 – 1 scale", "#3b82f6"),
        (col5, f"{int(report['avg_time_ms'])} ms",  "Avg Response Time",   "Per question", "#8b5cf6"),
    ]
    for col, value, label, target, color in metrics:
        with col:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-value' style='color:{color}'>{value}</div>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-target'>{target}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Domain breakdown
    st.markdown("#### Domain Breakdown — Citation Accuracy")
    if report.get("domain_breakdown"):
        domain_df = pd.DataFrame(
            [(d, v) for d, v in sorted(report["domain_breakdown"].items())],
            columns=["Domain", "Citation Accuracy (%)"],
        )
        st.bar_chart(domain_df.set_index("Domain"), color="#3b82f6", height=220)

    st.markdown("---")

    # Per-question table
    st.markdown("#### Per-Question Results")
    if report.get("rows"):
        df = pd.DataFrame(report["rows"])
        # Filter controls
        fcol1, fcol2 = st.columns([1, 3])
        with fcol1:
            domains = sorted(d for d in df["Domain"].unique().tolist() if d and str(d) != "nan")
            filter_domain = st.selectbox("Filter by domain", ["All"] + domains)
        with fcol2:
            filter_result = st.radio(
                "Show", ["All", "Citation failures", "Escalated"], horizontal=True
            )

        if filter_domain != "All":
            df = df[df["Domain"] == filter_domain]
        if filter_result == "Citation failures":
            df = df[df["Citation ✓"] == "❌"]
        elif filter_result == "Escalated":
            df = df[df["Escalated"] == "Yes"]

        st.dataframe(
            df[["ID", "Domain", "Question", "Answerable",
                "Citation ✓", "Escalated", "Confidence", "Level", "Time (ms)"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Question": st.column_config.TextColumn(width="large"),
                "Confidence": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.2f"
                ),
            },
        )

        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇ Download full results as CSV",
            data=csv,
            file_name="evaluation_results.csv",
            mime="text/csv",
        )


# ── Page layout ───────────────────────────────────────────────────────────────

st.markdown("# 📊 Evaluation Dashboard")
st.caption(
    "Runs the 60-question benchmark across all GxP domains. "
    "Each question calls the live pipeline — allow 5–10 minutes and API credits."
)

REPORT_PATH = Path(__file__).parent.parent / "reports" / "evaluation_report.json"

# Load existing report if present
cached_report = None
if REPORT_PATH.exists():
    try:
        cached_report = json.loads(REPORT_PATH.read_text())
    except Exception:
        pass

col_run, col_info = st.columns([1, 3])
with col_run:
    run_btn = st.button("▶  Run Evaluation", type="primary", use_container_width=True)
with col_info:
    if cached_report:
        st.info(
            f"Showing saved report from {cached_report.get('timestamp', 'unknown')}. "
            "Click **Run Evaluation** to generate a fresh one.",
            icon="ℹ️",
        )
    else:
        st.info("No saved report found. Click **Run Evaluation** to start.", icon="ℹ️")

if run_btn:
    st.markdown("---")
    st.markdown("#### Running… please wait")
    progress_bar = st.progress(0)
    status_text = st.empty()
    log_container = st.empty()

    agent = _load_agent()
    with st.spinner(""):
        report = _run_evaluation(agent, progress_bar, status_text, log_container)

    progress_bar.empty()
    status_text.empty()
    log_container.empty()
    st.success("Evaluation complete! Report saved to `reports/evaluation_report.json`")
    st.markdown("---")
    _show_report(report)

elif cached_report:
    st.markdown("---")
    _show_report(cached_report)
