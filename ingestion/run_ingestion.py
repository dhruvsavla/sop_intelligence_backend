"""
Run the full SOP ingestion pipeline:
  1. Load SOP files from data/sops/
  2. Parse into chunks
  3. Embed with sentence-transformers
  4. Ingest into ChromaDB
  5. Save ingestion_summary.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings
from ingestion.chroma_store import ChromaSOPStore
from ingestion.embedder import SOPEmbedder
from ingestion.parser import parse_sop_into_chunks

METADATA_PATH = Path(__file__).parent.parent / "sop_generator" / "sop_metadata.json"
SUMMARY_PATH = Path(__file__).parent.parent / "reports" / "ingestion_summary.json"


def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        print(f"⚠️  Metadata file not found: {METADATA_PATH}")
        print("   Run sop_generator/generate_sops.py first.")
        sys.exit(1)
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def main():
    print("═" * 60)
    print("  SOPAssist Ingestion Pipeline")
    print("═" * 60)

    # 1. Load metadata
    metadata_list = load_metadata()
    print(f"✅ Loaded metadata for {len(metadata_list)} SOPs")

    # 2. Parse all SOPs into chunks
    sop_dir = Path(settings.SOP_DATA_PATH)
    all_chunks: list[dict] = []
    skipped_files = 0

    for meta in metadata_list:
        file_path = meta.get("file_path")
        if not file_path or not Path(file_path).exists():
            # Try to locate by convention
            fname = f"{meta['sop_number']}_v{meta['version']}.txt"
            file_path = str(sop_dir / fname)

        if not Path(file_path).exists():
            print(f"  ⚠️  Missing SOP file: {file_path}")
            skipped_files += 1
            continue

        meta["is_current_version"] = True  # all generated SOPs are current
        chunks = parse_sop_into_chunks(file_path, meta)
        all_chunks.extend(chunks)

    print(f"✅ Parsed {len(all_chunks)} chunks from {len(metadata_list) - skipped_files} SOP files")
    if skipped_files:
        print(f"   ⚠️  Skipped {skipped_files} missing SOP files")

    if not all_chunks:
        print("❌ No chunks to ingest. Exiting.")
        sys.exit(1)

    # 3. Embed
    embedder = SOPEmbedder(settings.EMBEDDING_MODEL)
    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch = all_chunks[i : i + batch_size]
        vecs = embedder.embed_chunks(batch)
        embeddings.extend(vecs)

    print(f"✅ Generated {len(embeddings)} embeddings")

    # 4. Ingest into ChromaDB
    store = ChromaSOPStore(
        persist_directory=settings.CHROMA_DB_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )
    print(f"\nIngesting into ChromaDB at {settings.CHROMA_DB_PATH}...")
    store.ingest_chunks(all_chunks, embeddings)

    stats = store.get_collection_stats()
    print(f"✅ ChromaDB collection '{settings.CHROMA_COLLECTION_NAME}' now has {stats['count']} entries")
    print(f"   Domains: {', '.join(stats['domains'])}")
    print(f"   SOPs indexed: {len(stats['sop_numbers'])}")

    # 5. Save summary
    chunks_per_domain: dict[str, int] = {}
    for c in all_chunks:
        d = c.get("domain", "UNKNOWN")
        chunks_per_domain[d] = chunks_per_domain.get(d, 0) + 1

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_sops": len(metadata_list) - skipped_files,
        "total_chunks": len(all_chunks),
        "chunks_per_domain": chunks_per_domain,
        "collection_stats": stats,
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n✅ Ingestion summary saved to {SUMMARY_PATH}")
    print("═" * 60)


if __name__ == "__main__":
    main()
