"""
SOPAssist — SOP Viewer
Displays the full text of a specific SOP with the referenced section highlighted.
Opened via "View SOP ↗" links in the Retrieved Sources panel.

URL: /View_SOP?sop=SOP-GMP-001&section=4.2
"""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from ingestion.chroma_store import ChromaSOPStore

st.set_page_config(
    page_title="SOP Viewer — SOPAssist",
    page_icon="⚕",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 15px; }
#MainMenu, footer { visibility: hidden; height: 0; }
[data-testid="stAppViewContainer"] { background: #f4f7f9; }
.main .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1423 0%, #16263f 100%);
    border-right: 1px solid #1e3150;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.sop-header {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 6px rgba(15,23,42,0.04);
}
.chunk-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.2s ease;
}
.chunk-card.highlighted {
    background: #eff6ff;
    border-color: #93c5fd;
    border-left: 4px solid #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08);
}
.chunk-section-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.chunk-section-label.highlighted { color: #1d4ed8; }
.chunk-text {
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.7;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# ── Cached store ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_store():
    return ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


def _section_sort_key(section_num: str) -> tuple:
    try:
        return tuple(int(x) for x in section_num.split("."))
    except (ValueError, AttributeError):
        return (0,)


# ── Resolve target SOP (session state from chat button → query params fallback) ─
# The "View SOP ↗" button in the chat uses st.switch_page, which preserves the
# Streamlit session so chat history stays intact when the user navigates back.

_req = st.session_state.get("view_sop_request", {})
sop_number = _req.get("sop") or st.query_params.get("sop", "")
target_section = _req.get("section") or st.query_params.get("section", "")

col_back, _ = st.columns([2, 8])
with col_back:
    if st.button("← Back to Chat", use_container_width=True):
        st.switch_page("streamlit_app.py")

if not sop_number:
    st.info("No SOP selected. Open this page via a **View SOP ↗** button in the chat.")
    st.stop()

# ── Fetch all chunks for this SOP ─────────────────────────────────────────────

store = _load_store()

with st.spinner(f"Loading {sop_number}…"):
    result = store.get(
        where={"sop_number": sop_number},
        limit=1000,
        include=["documents", "metadatas"],
    )

documents = result.get("documents") or []
metadatas = result.get("metadatas") or []

if not documents:
    st.error(f"No content found for **{sop_number}**. It may not be in the knowledge base yet.")
    st.stop()

chunks = [
    {"text": doc, **meta}
    for doc, meta in zip(documents, metadatas)
    if doc and meta
]
chunks.sort(key=lambda c: _section_sort_key(c.get("section_number", "0")))

# ── SOP header ────────────────────────────────────────────────────────────────

DOMAIN_COLORS = {
    "GMP": ("#d1fae5", "#065f46"),
    "GCP": ("#dbeafe", "#1e3a8a"),
    "GLP": ("#fef3c7", "#78350f"),
    "PV":  ("#fce7f3", "#831843"),
    "DI":  ("#ede9fe", "#4c1d95"),
}

first = chunks[0]
sop_title    = first.get("sop_title", sop_number)
version      = first.get("version", "")
domain       = first.get("domain", "")
eff_date     = first.get("effective_date", "")
approver     = first.get("approver", "")

domain_bg, domain_fg = DOMAIN_COLORS.get(domain, ("#f1f5f9", "#475569"))

meta_items = " · ".join(filter(None, [
    f"v{version}" if version else "",
    f"Effective {eff_date}" if eff_date else "",
    f"Approved by {approver}" if approver else "",
    f"{len(chunks)} chunks",
]))

st.markdown(
    f"<div class='sop-header'>"
    f"  <div style='display:flex;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:6px'>"
    f"    <h2 style='margin:0;font-size:1.4rem;font-weight:700;color:#0f172a;flex:1'>{sop_title}</h2>"
    f"    <span style='background:{domain_bg};color:{domain_fg};border-radius:4px;"
    f"      padding:3px 10px;font-size:0.68rem;font-weight:800;text-transform:uppercase;"
    f"      letter-spacing:0.06em;white-space:nowrap'>{domain}</span>"
    f"  </div>"
    f"  <div style='font-size:0.78rem;color:#64748b;font-weight:500'>"
    f"    <span style='font-weight:700;color:#334155'>{sop_number}</span>"
    f"    {' · ' + meta_items if meta_items else ''}"
    f"  </div>"
    f"</div>",
    unsafe_allow_html=True,
)

if target_section:
    st.markdown(
        f"<div style='font-size:0.8rem;color:#1d4ed8;font-weight:600;"
        f"margin-bottom:1rem;padding:6px 10px;background:#eff6ff;"
        f"border:1px solid #bfdbfe;border-radius:6px;display:inline-block'>"
        f"Showing Section {target_section}</div>",
        unsafe_allow_html=True,
    )

# ── Render chunks ─────────────────────────────────────────────────────────────

scroll_id = None
html_parts = []

for i, chunk in enumerate(chunks):
    is_target = bool(target_section) and chunk.get("section_number") == target_section
    cid = f"sop-chunk-{i}"
    if is_target and scroll_id is None:
        scroll_id = cid

    sec_num   = chunk.get("section_number", "")
    sec_title = chunk.get("section_title", f"Section {sec_num}")
    label     = f"§{sec_num}  {sec_title}" if sec_num else sec_title
    text      = chunk.get("text", "").replace("<", "&lt;").replace(">", "&gt;")

    card_class   = "chunk-card highlighted" if is_target else "chunk-card"
    label_class  = "chunk-section-label highlighted" if is_target else "chunk-section-label"

    html_parts.append(
        f"<div id='{cid}' class='{card_class}'>"
        f"  <div class='{label_class}'>{label}</div>"
        f"  <div class='chunk-text'>{text}</div>"
        f"</div>"
    )

st.markdown("\n".join(html_parts), unsafe_allow_html=True)

# ── Auto-scroll to highlighted section ───────────────────────────────────────

if scroll_id:
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            var el = window.parent.document.getElementById('{scroll_id}');
            if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}, 700);
        </script>
        """,
        height=0,
    )
