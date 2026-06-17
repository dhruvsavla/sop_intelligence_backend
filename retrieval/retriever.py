"""
Version-aware retrieval from ChromaDB.

Primary path  : LlamaIndex ChromaVectorStore (richer metadata-filter API).
Fallback path : direct chromadb.Collection.query() when llama-index packages
                are not installed — identical RetrievalResult output either way.
"""
import os
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

    if section == "0":
        section_title = meta.get("section_title", "General")
        return f"[{sop_num} v{version}, {section_title}]"

    return f"[{sop_num} v{version}, Section {section}]"


def _make_fastembed_adapter(fastembed_model):
    """
    Build a LlamaIndex-compatible embedding model that delegates to fastembed.
    This avoids the torch/sentence-transformers dependency on Python 3.13.
    """
    from llama_index.core.embeddings import BaseEmbedding

    class _FastEmbedAdapter(BaseEmbedding):
        class Config:
            arbitrary_types_allowed = True

        def _get_query_embedding(self, query: str) -> list[float]:
            return list(fastembed_model.embed([query]))[0].tolist()

        def _get_text_embedding(self, text: str) -> list[float]:
            return list(fastembed_model.embed([text]))[0].tolist()

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        async def _aget_text_embedding(self, text: str) -> list[float]:
            return self._get_text_embedding(text)

    return _FastEmbedAdapter()


def _init_llamaindex(collection, embedder):
    """
    Wrap an existing ChromaDB collection in a LlamaIndex VectorStoreIndex.
    Returns the index, or None if the llama-index packages are not installed.
    """
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    try:
        from llama_index.core import Settings, VectorStoreIndex
        from llama_index.core.storage.storage_context import StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore

        # Attach the already-loaded embedding model to LlamaIndex Settings.
        if embedder.mode == "fastembed":
            # fastembed.TextEmbedding — wrap with a thin adapter (no torch needed)
            Settings.embed_model = _make_fastembed_adapter(embedder._impl.model)
        elif embedder.mode == "sentence_transformers":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise RuntimeError("BM25 mode: LlamaIndex vector search unavailable")

        Settings.llm = None  # Claude is called explicitly in qa_agent.py

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        print("✅ LlamaIndex retriever initialised")
        return index
    except Exception as exc:
        print(f"LlamaIndex unavailable — falling back to direct ChromaDB: {exc}")
        return None


class VersionAwareRetriever:
    def __init__(self, chroma_store: ChromaSOPStore, embedder: SOPEmbedder):
        self.store = chroma_store
        self.embedder = embedder

        # Resolve the raw chromadb Collection for LlamaIndex wrapping
        raw = getattr(chroma_store, "collection", None)
        if raw is None:
            raw = getattr(getattr(chroma_store, "_store", None), "collection", None)
        self._index = _init_llamaindex(raw, embedder) if raw is not None else None

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        domain_filter: Optional[str] = None,
        sop_filter: Optional[str] = None,
        include_superseded: bool = False,
    ) -> list[RetrievalResult]:
        if self._index is not None:
            return self._retrieve_llamaindex(
                query, n_results, domain_filter, sop_filter, include_superseded
            )
        return self._retrieve_direct(
            query, n_results, domain_filter, sop_filter, include_superseded
        )

    # ── LlamaIndex path ───────────────────────────────────────────────────────

    def _retrieve_llamaindex(
        self,
        query: str,
        n_results: int,
        domain_filter: Optional[str],
        sop_filter: Optional[str],
        include_superseded: bool,
    ) -> list[RetrievalResult]:
        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        filter_list: list[MetadataFilter] = []
        if not include_superseded:
            filter_list.append(
                MetadataFilter(key="is_current_version", value="true", operator=FilterOperator.EQ)
            )
        if domain_filter:
            filter_list.append(
                MetadataFilter(key="domain", value=domain_filter.upper(), operator=FilterOperator.EQ)
            )
        if sop_filter:
            filter_list.append(
                MetadataFilter(key="sop_number", value=sop_filter.upper(), operator=FilterOperator.EQ)
            )

        filters = MetadataFilters(filters=filter_list) if filter_list else None
        retriever = self._index.as_retriever(
            similarity_top_k=n_results * 2,
            filters=filters,
        )
        nodes = retriever.retrieve(query)

        results: list[RetrievalResult] = []
        for node in nodes:
            meta = node.metadata or {}
            score = float(node.score) if node.score is not None else 0.0
            score = round(max(0.0, min(1.0, score)), 4)
            results.append(RetrievalResult(
                chunk_id=node.node_id,
                text=node.get_content(),
                similarity_score=score,
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

        results = self._deduplicate(results)
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:n_results]

    # ── Direct ChromaDB fallback ──────────────────────────────────────────────

    def _retrieve_direct(
        self,
        query: str,
        n_results: int,
        domain_filter: Optional[str],
        sop_filter: Optional[str],
        include_superseded: bool,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(query)
        where = self._build_where(domain_filter, sop_filter, include_superseded)
        raw = self.store.query(
            query_embedding=query_embedding,
            n_results=max(n_results * 2, 10),
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
            similarity = 1.0 / (1.0 + dist)
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
