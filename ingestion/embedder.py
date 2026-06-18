"""
Embedding layer with priority fallback chain:
  1. fastembed   — ONNX-based, no torch, works on Python 3.13 (preferred)
  2. sentence-transformers — torch-based, works on Python 3.12 with torch wheels
  3. BM25PassthroughEmbedder — pure Python, zero-dependency last resort

Context-injection (fastembed + sentence-transformers): chunk texts are prefixed
with SOP title and section title before embedding, matching the asymmetric
retrieval pattern expected by the cosine-space ChromaDB collection.
Query embeddings are always plain text (no prefix) — intentional.
"""

class BM25PassthroughEmbedder:
    """
    Zero-dependency embedder for environments without onnxruntime/torch.
    Pairs with BM25SOPStore — the 'embedding' is just a tagged query string.
    """

    def __init__(self, model_name: str = "bm25"):
        print("No neural embedding available — using BM25 retrieval (pure Python)")
        print("✅ BM25 embedder ready (no ML model needed)")

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        return [[0.0] * 384 for _ in chunks]

    def embed_query(self, query: str) -> "BM25QueryVector":
        return BM25QueryVector(query)


class BM25QueryVector:
    """Thin wrapper that carries the raw query string for BM25SOPStore.query()."""

    def __init__(self, query: str):
        self._raw_query = query

    def tolist(self):
        return self  # BM25Store checks for _raw_query attr, not a list


def _load_fastembed_embedder(model_name: str):
    from fastembed import TextEmbedding

    class FastEmbedEmbedder:
        def __init__(self, name: str):
            print(f"Loading embedding model via fastembed: {name}...")
            self.model = TextEmbedding(name)
            print("✅ Embedding model loaded (dim=384)")

        def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
            texts = [
                f"Document: {c.get('sop_title', '')} | Section: {c.get('section_title', '')}\n{c['text']}"
                for c in chunks
            ]
            return [e.tolist() for e in self.model.embed(texts)]

        def embed_query(self, query: str) -> list[float]:
            return list(self.model.embed([query]))[0].tolist()

    return FastEmbedEmbedder(model_name)


def _load_sentence_transformer_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    class SentenceTransformerEmbedder:
        def __init__(self, name: str):
            print(f"Loading embedding model via sentence-transformers: {name}...")
            self.model = SentenceTransformer(name)
            print("✅ Embedding model loaded (dim=384)")

        def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
            texts = [
                f"Document: {c.get('sop_title', '')} | Section: {c.get('section_title', '')}\n{c['text']}"
                for c in chunks
            ]
            return self.model.encode(texts, show_progress_bar=True).tolist()

        def embed_query(self, query: str) -> list[float]:
            return self.model.encode([query])[0].tolist()

    return SentenceTransformerEmbedder(model_name)


class SOPEmbedder:
    """
    Auto-detecting embedder — tries fastembed, then sentence-transformers, then BM25.
    The public interface (embed_chunks / embed_query) is identical in every mode.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        for loader, label in [
            (_load_fastembed_embedder, "fastembed"),
            (_load_sentence_transformer_embedder, "sentence_transformers"),
        ]:
            try:
                self._impl = loader(model_name)
                self.mode = label
                return
            except (ImportError, Exception):
                pass
        self._impl = BM25PassthroughEmbedder(model_name)
        self.mode = "bm25"

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        return self._impl.embed_chunks(chunks)

    def embed_query(self, query: str):
        return self._impl.embed_query(query)
