"""
BM25-based vector store — zero onnxruntime/torch dependency.
Works on Python 3.12, 3.13, and 3.14.

Replaces chromadb + fastembed for environments where onnxruntime
wheels are unavailable (e.g. Python 3.14).
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


@dataclass
class BM25Chunk:
    chunk_id: str
    text: str
    sop_number: str
    sop_title: str
    version: str
    effective_date: str
    domain: str
    section_number: str
    section_title: str
    is_current_version: str  # "true"/"false" string to match ChromaDB convention
    approver: str
    keywords: str


class BM25SOPStore:
    """
    Drop-in replacement for ChromaSOPStore using BM25 text ranking.
    Persists to <persist_directory>/bm25_store.json.
    """

    def __init__(self, persist_directory: str, collection_name: str):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._store_path = self.persist_directory / f"{collection_name}_bm25.json"
        self._chunks: list[BM25Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._load()

    def _load(self):
        if self._store_path.exists():
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._chunks = [BM25Chunk(**d) for d in data]
            self._rebuild_index()

    def _rebuild_index(self):
        if self._chunks:
            corpus = [_tokenize(c.text) for c in self._chunks]
            self._bm25 = BM25Okapi(corpus)

    def _save(self):
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(
            json.dumps([asdict(c) for c in self._chunks], indent=2),
            encoding="utf-8",
        )

    def get_or_create_collection(self):
        return self  # compatibility shim

    def ingest_chunks(self, chunks: list[dict], embeddings=None):
        existing_ids = {c.chunk_id for c in self._chunks}
        for c in chunks:
            if c["chunk_id"] not in existing_ids:
                self._chunks.append(BM25Chunk(
                    chunk_id=c.get("chunk_id", ""),
                    text=c.get("text", ""),
                    sop_number=c.get("sop_number", ""),
                    sop_title=c.get("sop_title", ""),
                    version=c.get("version", ""),
                    effective_date=c.get("effective_date", ""),
                    domain=c.get("domain", ""),
                    section_number=c.get("section_number", ""),
                    section_title=c.get("section_title", ""),
                    is_current_version="true" if c.get("is_current_version", True) else "false",
                    approver=c.get("approver", ""),
                    keywords=c.get("keywords", ""),
                ))
        self._rebuild_index()
        self._save()

    def collection_exists_and_populated(self) -> bool:
        return len(self._chunks) > 0

    def get_collection_stats(self) -> dict:
        domains = sorted({c.domain for c in self._chunks if c.domain})
        sop_numbers = sorted({c.sop_number for c in self._chunks if c.sop_number})
        return {"count": len(self._chunks), "domains": domains, "sop_numbers": sop_numbers}

    def query(self, query_embedding, n_results: int = 5, where: dict = None) -> dict:
        if not self._chunks or self._bm25 is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # The query_embedding here is actually the query string for BM25 mode
        # (SOPEmbedder.embed_query returns the raw text in BM25 mode)
        query_text = getattr(query_embedding, "_raw_query", None)
        if query_text is None:
            # Fallback: treat embedding as meaningless, return first n results
            candidates = self._apply_filter(self._chunks, where)
            top = candidates[:n_results]
            return self._format_results(top, [0.5] * len(top))

        tokens = _tokenize(query_text)
        all_scores = self._bm25.get_scores(tokens)

        # Apply metadata filter
        candidates_with_scores = [
            (chunk, all_scores[i])
            for i, chunk in enumerate(self._chunks)
        ]
        if where:
            candidates_with_scores = [
                (c, s) for c, s in candidates_with_scores
                if self._matches_filter(c, where)
            ]

        # Sort by score descending
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        top = candidates_with_scores[:n_results]

        # Normalize scores to [0, 1] — treat max_score as 1.0
        max_score = top[0][1] if top and top[0][1] > 0 else 1.0
        distances = [max(0.0, 1.0 - (s / max_score)) for _, s in top]

        return self._format_results([c for c, _ in top], distances)

    def _apply_filter(self, chunks: list[BM25Chunk], where: dict) -> list[BM25Chunk]:
        if not where:
            return chunks
        return [c for c in chunks if self._matches_filter(c, where)]

    def _matches_filter(self, chunk: BM25Chunk, where: dict) -> bool:
        if "$and" in where:
            return all(self._matches_filter(chunk, cond) for cond in where["$and"])
        for key, value in where.items():
            if key.startswith("$"):
                continue
            chunk_val = getattr(chunk, key, None)
            if chunk_val != value:
                return False
        return True

    def _format_results(self, chunks: list[BM25Chunk], distances: list[float]) -> dict:
        return {
            "ids": [[c.chunk_id for c in chunks]],
            "documents": [[c.text for c in chunks]],
            "metadatas": [[{
                "sop_number": c.sop_number,
                "sop_title": c.sop_title,
                "version": c.version,
                "effective_date": c.effective_date,
                "domain": c.domain,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "is_current_version": c.is_current_version,
                "approver": c.approver,
                "keywords": c.keywords,
            } for c in chunks]],
            "distances": [distances],
        }

    # ChromaDB compatibility — get() used by /api/sops/{sop_number} endpoint
    def get(self, where: dict = None, limit: int = None, include=None) -> dict:
        candidates = self._apply_filter(self._chunks, where)
        if limit:
            candidates = candidates[:limit]
        return {
            "metadatas": [{
                "sop_number": c.sop_number,
                "sop_title": c.sop_title,
                "version": c.version,
                "effective_date": c.effective_date,
                "domain": c.domain,
                "section_number": c.section_number,
                "section_title": c.section_title,
                "is_current_version": c.is_current_version,
                "approver": c.approver,
                "keywords": c.keywords,
            } for c in candidates],
        }

    def count(self) -> int:
        return len(self._chunks)
