"""
SOPAssist — Upload SOP
Upload a PDF SOP and ingest it into the knowledge base.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from ingestion.chroma_store import ChromaSOPStore
from ingestion.embedder import SOPEmbedder
from ingestion.pdf_parser import parse_pdf_into_chunks

st.set_page_config(
    page_title="Upload SOP — SOPAssist",
    page_icon="⚕",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 15px; }
#MainMenu, footer { visibility: hidden; height: 0; }
[data-testid="stAppViewContainer"] { background: #f4f7f9; }
.main .block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 4rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1423 0%, #16263f 100%);
    border-right: 1px solid #1e3150;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.chunk-preview {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.75rem;
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_store_and_embedder():
    embedder = SOPEmbedder(settings.EMBEDDING_MODEL)
    store = ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    return store, embedder


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown(
    "<h2 style='font-size:1.6rem;font-weight:700;color:#0f172a;margin-bottom:0.25rem'>"
    "Upload SOP</h2>"
    "<p style='color:#64748b;font-size:0.9rem;margin-bottom:2rem'>"
    "Upload a PDF Standard Operating Procedure to add it to the knowledge base. "
    "It will be chunked, embedded, and stored so it can be queried immediately.</p>",
    unsafe_allow_html=True,
)

# ── Current KB stats ──────────────────────────────────────────────────────────

try:
    store, _ = _load_store_and_embedder()
    stats = store.get_collection_stats()
    n_sops = len(stats.get("sop_numbers", []))
    n_chunks = stats.get("count", 0)
    st.info(f"Knowledge base currently contains **{n_sops} SOPs** and **{n_chunks:,} chunks**.")
except Exception:
    pass

st.divider()

# ── Upload form ───────────────────────────────────────────────────────────────

DOMAIN_OPTIONS = ["GMP", "GCP", "GLP", "PV", "DI", "Other"]

with st.form("upload_form", clear_on_submit=False):
    st.markdown("#### SOP Metadata")

    col1, col2 = st.columns(2)
    with col1:
        sop_number = st.text_input("SOP Number *", placeholder="e.g. SOP-GMP-050")
        version = st.text_input("Version *", value="1.0", placeholder="e.g. 1.0")
        domain = st.selectbox("Domain *", DOMAIN_OPTIONS)
        approver = st.text_input("Approver", placeholder="e.g. Quality Manager")
    with col2:
        title = st.text_input("SOP Title *", placeholder="e.g. Equipment Cleaning Procedure")
        effective_date = st.text_input(
            "Effective Date", placeholder="e.g. 2024-01-15"
        )
        keywords = st.text_input(
            "Keywords (comma-separated)",
            placeholder="e.g. cleaning, equipment, validation",
        )

    st.markdown("#### PDF File")
    pdf_file = st.file_uploader("Upload PDF *", type=["pdf"], label_visibility="collapsed")

    submitted = st.form_submit_button(
        "Ingest into Knowledge Base", use_container_width=True, type="primary"
    )

# ── Process submission ────────────────────────────────────────────────────────

if submitted:
    errors = []
    if not sop_number.strip():
        errors.append("SOP Number is required.")
    if not title.strip():
        errors.append("SOP Title is required.")
    if not version.strip():
        errors.append("Version is required.")
    if not pdf_file:
        errors.append("Please upload a PDF file.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        metadata = {
            "sop_number": sop_number.strip(),
            "title": title.strip(),
            "version": version.strip(),
            "effective_date": effective_date.strip(),
            "domain": domain if domain != "Other" else "",
            "approver": approver.strip(),
            "keywords": keywords.strip(),
        }

        try:
            with st.spinner("Extracting text from PDF..."):
                chunks = parse_pdf_into_chunks(pdf_file, metadata)

            if not chunks:
                st.error(
                    "No text could be extracted from this PDF. "
                    "The file may be a scanned image — only text-based PDFs are supported."
                )
            else:
                store, embedder = _load_store_and_embedder()

                progress = st.progress(0, text=f"Embedding {len(chunks)} chunks...")
                embeddings = embedder.embed_chunks(chunks)

                progress.progress(60, text="Storing in knowledge base...")
                store.ingest_chunks(chunks, embeddings)
                progress.progress(100, text="Done!")

                st.success(
                    f"**{sop_number}** — {title} ingested successfully.\n\n"
                    f"**{len(chunks)} chunks** added across {len({c['section_number'] for c in chunks})} pages."
                )

                # Clear cached stats so sidebar reflects new count on next page load
                st.cache_data.clear()

                with st.expander("Preview extracted chunks", expanded=False):
                    for i, chunk in enumerate(chunks[:5]):
                        preview = chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else "")
                        st.markdown(
                            f"<div class='chunk-preview'>"
                            f"<strong>{chunk['section_title']}</strong> &nbsp;·&nbsp; "
                            f"<span style='color:#94a3b8;font-size:0.75rem'>{len(chunk['text'])} chars</span>"
                            f"<br><br>{preview}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    if len(chunks) > 5:
                        st.caption(f"… and {len(chunks) - 5} more chunks")

        except ImportError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
