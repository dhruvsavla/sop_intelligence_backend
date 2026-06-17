"""
Vector store factory: returns ChromaSOPStore (fastembed mode) or
BM25SOPStore (pure-Python fallback mode), automatically selected based
on whether chromadb + onnxruntime are importable.
"""
import os


def _build_chroma_store(persist_directory: str, collection_name: str):
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    import chromadb

    class ChromaSOPStore:
        def __init__(self):
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.get_or_create_collection()

        def get_or_create_collection(self):
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        def ingest_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
            if not chunks:
                return
            ids = [c["chunk_id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [
                {
                    "sop_number": c.get("sop_number", ""),
                    "sop_title": c.get("sop_title", ""),
                    "version": c.get("version", ""),
                    "effective_date": c.get("effective_date", ""),
                    "domain": c.get("domain", ""),
                    "section_number": c.get("section_number", ""),
                    "section_title": c.get("section_title", ""),
                    "is_current_version": "true" if c.get("is_current_version", True) else "false",
                    "approver": c.get("approver", ""),
                    "keywords": c.get("keywords", ""),
                }
                for c in chunks
            ]
            for i in range(0, len(ids), 500):
                self.collection.upsert(
                    ids=ids[i : i + 500],
                    embeddings=embeddings[i : i + 500],
                    documents=documents[i : i + 500],
                    metadatas=metadatas[i : i + 500],
                )

        def collection_exists_and_populated(self) -> bool:
            try:
                return self.collection.count() > 0
            except Exception:
                return False

        def get_collection_stats(self) -> dict:
            count = self.collection.count()
            if count == 0:
                return {"count": 0, "domains": [], "sop_numbers": []}
            results = self.collection.get(limit=min(count, 5000), include=["metadatas"])
            domains, sop_numbers = set(), set()
            for m in results.get("metadatas", []):
                if m:
                    domains.add(m.get("domain", ""))
                    sop_numbers.add(m.get("sop_number", ""))
            return {
                "count": count,
                "domains": sorted(d for d in domains if d),
                "sop_numbers": sorted(s for s in sop_numbers if s),
            }

        def query(self, query_embedding, n_results: int = 5, where: dict = None) -> dict:
            kwargs = dict(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            if where:
                kwargs["where"] = where
            return self.collection.query(**kwargs)

        # For /api/sops/{sop_number} endpoint
        def get(self, where: dict = None, limit: int = None, include=None) -> dict:
            kwargs = {"include": include or ["metadatas"]}
            if where:
                kwargs["where"] = where
            if limit:
                kwargs["limit"] = limit
            return self.collection.get(**kwargs)

        def count(self) -> int:
            return self.collection.count()

    return ChromaSOPStore()


def build_sop_store(persist_directory: str, collection_name: str, embedder_mode: str = "auto"):
    """
    Return the best available store for the current environment.

    embedder_mode: "auto" | "chroma" | "bm25"
    """
    if embedder_mode == "bm25":
        from ingestion.bm25_store import BM25SOPStore
        return BM25SOPStore(persist_directory, collection_name)

    if embedder_mode == "chroma":
        return _build_chroma_store(persist_directory, collection_name)

    # Auto-detect
    try:
        store = _build_chroma_store(persist_directory, collection_name)
        return store
    except (ImportError, Exception):
        from ingestion.bm25_store import BM25SOPStore
        return BM25SOPStore(persist_directory, collection_name)


# Backward-compat alias (used by run_ingestion.py and evaluator.py)
class ChromaSOPStore:
    """Thin proxy that delegates to whichever store build_sop_store returns."""

    def __init__(self, persist_directory: str, collection_name: str):
        self._store = build_sop_store(persist_directory, collection_name)
        # Expose collection attribute for direct access in main.py
        self.collection = getattr(self._store, "collection", self._store)

    def __getattr__(self, name):
        return getattr(self._store, name)
