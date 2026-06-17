"""
Embedding layer with automatic fallback:
  - Primary: fastembed (onnxruntime-based, requires Python <=3.13)
  - Fallback: BM25PassthroughEmbedder (pure Python, works on Python 3.14)

The BM25 fallback stores the raw query string in the returned object so
BM25SOPStore can tokenize it directly during retrieval.
"""


class BM25PassthroughEmbedder:
    """
    Zero-dependency embedder for environments without onnxruntime/torch.
    Pairs with BM25SOPStore — the 'embedding' is just a tagged query string.
    """

    def __init__(self, model_name: str = "bm25"):
        print("onnxruntime/fastembed unavailable — using BM25 retrieval (pure Python)")
        print("✅ BM25 embedder ready (no ML model needed)")

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        # BM25Store ignores embeddings during ingest; return dummies
        return [[0.0] * 384 for _ in chunks]

    def embed_query(self, query: str) -> "BM25QueryVector":
        return BM25QueryVector(query)


class BM25QueryVector:
    """Thin wrapper that carries the raw query string for BM25SOPStore.query()."""

    def __init__(self, query: str):
        self._raw_query = query

    def tolist(self):
        return self  # BM25Store checks for _raw_query attr, not a list


FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_fastembed_embedder(model_name: str):
    from fastembed import TextEmbedding

    class FastEmbedEmbedder:
        def __init__(self):
            print(f"Loading embedding model via fastembed: {FASTEMBED_MODEL}...")
            self.model = TextEmbedding(FASTEMBED_MODEL)
            print("✅ Embedding model loaded (dim=384)")

        def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
            texts = [c["text"] for c in chunks]
            return [e.tolist() for e in self.model.embed(texts)]

        def embed_query(self, query: str) -> list[float]:
            return list(self.model.embed([query]))[0].tolist()

    return FastEmbedEmbedder()


class SOPEmbedder:
    """
    Auto-detecting embedder: tries fastembed first, falls back to BM25.
    The interface is identical either way.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self._impl = _load_fastembed_embedder(model_name)
            self.mode = "fastembed"
        except (ImportError, Exception):
            self._impl = BM25PassthroughEmbedder(model_name)
            self.mode = "bm25"

    def embed_chunks(self, chunks: list[dict]) -> list[list[float]]:
        return self._impl.embed_chunks(chunks)

    def embed_query(self, query: str):
        return self._impl.embed_query(query)
