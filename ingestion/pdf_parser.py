"""
Parse uploaded PDF files into chunks compatible with the ingestion pipeline.
Uses pdfplumber for text extraction; splits by page, then by size with overlap.
"""

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 1200
OVERLAP_SIZE = 100


def _split_text(text: str, base_id: str, meta: dict, seen_ids: set) -> list[dict]:
    parts = []
    start = 0
    part_num = 1
    while start < len(text):
        end = start + MAX_CHUNK_SIZE
        segment = text[start:end]
        cid = f"{base_id}_c{part_num}"
        while cid in seen_ids:
            part_num += 1
            cid = f"{base_id}_c{part_num}"
        seen_ids.add(cid)
        parts.append({**meta, "chunk_id": cid, "text": segment})
        if end >= len(text):
            break
        start = end - OVERLAP_SIZE
        part_num += 1
    return parts


def parse_pdf_into_chunks(pdf_file, metadata: dict) -> list[dict]:
    """
    Extract text from a PDF file object and return ingestion-compatible chunks.

    pdf_file: file-like object or path accepted by pdfplumber.open()
    metadata: dict with keys sop_number, title, version, effective_date, domain, approver, keywords
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF ingestion. Run: pip install pdfplumber"
        )

    sop_number = metadata.get("sop_number", "UPLOADED")
    version = metadata.get("version", "1.0")

    base_meta = {
        "sop_number": sop_number,
        "sop_title": metadata.get("title", "Uploaded SOP"),
        "version": version,
        "effective_date": metadata.get("effective_date", ""),
        "domain": metadata.get("domain", ""),
        "approver": metadata.get("approver", ""),
        "keywords": metadata.get("keywords", ""),
        "is_current_version": True,
    }

    chunks: list[dict] = []
    seen_ids: set[str] = set()
    pending_text = ""
    pending_page = 1

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            # Accumulate tiny pages into the next page's text
            if pending_text:
                page_text = pending_text + "\n" + page_text
                page_num_label = pending_page
                pending_text = ""
            else:
                page_num_label = page_num

            meta = {
                **base_meta,
                "section_number": str(page_num_label),
                "section_title": f"Page {page_num_label}",
            }
            base_id = f"{sop_number}_v{version}_p{page_num_label}"

            if len(page_text) < MIN_CHUNK_SIZE:
                pending_text = page_text
                pending_page = page_num_label
                continue

            if len(page_text) <= MAX_CHUNK_SIZE:
                cid = base_id
                if cid in seen_ids:
                    cid = f"{base_id}_2"
                seen_ids.add(cid)
                chunks.append({**meta, "chunk_id": cid, "text": page_text})
            else:
                chunks.extend(_split_text(page_text, base_id, meta, seen_ids))

    # Flush any remaining small trailing page
    if pending_text and len(pending_text) >= MIN_CHUNK_SIZE:
        meta = {
            **base_meta,
            "section_number": str(pending_page),
            "section_title": f"Page {pending_page}",
        }
        cid = f"{sop_number}_v{version}_p{pending_page}_tail"
        seen_ids.add(cid)
        chunks.append({**meta, "chunk_id": cid, "text": pending_text})

    return chunks
