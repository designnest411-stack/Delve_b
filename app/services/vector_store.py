"""
Delve Embedding Service
────────────────────────
MiniLM-L6 embedding function for generating 384-dimension vectors.
Vectors are stored and queried via Supabase pgvector — ChromaDB is not used.
"""

import logging
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger("delve.embeddings")


class EmbeddingService:
    """Wraps the sentence-transformers MiniLM model for 384-dim embeddings."""

    def __init__(self):
        # DefaultEmbeddingFunction is the all-MiniLM-L6-v2 model bundled with
        # chromadb's utils — no ChromaDB client or collection is created here.
        self._fn = DefaultEmbeddingFunction()
        logger.info("MiniLM embedding function loaded")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate 384-dimension embeddings for a list of text strings."""
        if not texts:
            return []
        return [list(map(float, vec)) for vec in self._fn(texts)]


# Singleton — imported as `vector_store` for backwards compat with call sites
vector_store = EmbeddingService()
