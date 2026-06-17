"""
Version-aware retrieval from ChromaDB.
"""
from dataclasses import dataclass
from typing import Optional

from ingestion.chroma_store import ChromaSOPStore
from ingestion.embedder import SOPEmbedder


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    similarity_score: float
    sop_number: str
    sop_title: str
    version: str
    effective_date: str
    domain: str
    section_number: str
    section_title: str
    is_current_version: bool
    citation: str


def _build_citation(meta: dict) -> str:
    sop_num = meta.get("sop_number", "SOP-???")
    version = meta.get("version", "?.?")
    section = meta.get("section_number", "?")
    return f"[{sop_num} v{version}, Section {section}]"


class VersionAwareRetriever:
    def __init__(self, chroma_store: ChromaSOPStore, embedder: SOPEmbedder):
        self.store = chroma_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        domain_filter: Optional[str] = None,
        sop_filter: Optional[str] = None,
        include_superseded: bool = False,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)

        where = self._build_where(domain_filter, sop_filter, include_superseded)

        raw = self.store.query(
            query_embedding=query_embedding,
            n_results=max(n_results * 2, 10),  # over-fetch then deduplicate
            where=where,
        )

        results = self._parse_results(raw)
        results = self._deduplicate(results)
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:n_results]

    def _build_where(
        self,
        domain_filter: Optional[str],
        sop_filter: Optional[str],
        include_superseded: bool,
    ) -> Optional[dict]:
        conditions = []
        if not include_superseded:
            conditions.append({"is_current_version": "true"})
        if domain_filter:
            conditions.append({"domain": domain_filter.upper()})
        if sop_filter:
            conditions.append({"sop_number": sop_filter.upper()})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _parse_results(self, raw: dict) -> list[RetrievalResult]:
        results = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, 1.0 - dist)
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=text or "",
                similarity_score=round(similarity, 4),
                sop_number=meta.get("sop_number", ""),
                sop_title=meta.get("sop_title", ""),
                version=meta.get("version", ""),
                effective_date=meta.get("effective_date", ""),
                domain=meta.get("domain", ""),
                section_number=meta.get("section_number", ""),
                section_title=meta.get("section_title", ""),
                is_current_version=meta.get("is_current_version", "true") == "true",
                citation=_build_citation(meta),
            ))
        return results

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        seen: dict[str, RetrievalResult] = {}
        for r in results:
            key = f"{r.sop_number}_{r.section_number}"
            if key not in seen or r.similarity_score > seen[key].similarity_score:
                seen[key] = r
        return list(seen.values())
