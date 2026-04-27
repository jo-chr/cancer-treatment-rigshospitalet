"""Wrap the Chroma store with a small retrieval helper."""
from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return Chroma(
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
        collection_name="rigshospitalet",
    )


def retrieve(query: str, k: int | None = None) -> list[tuple[Document, float]]:
    """Return (document, similarity) pairs above the configured threshold.

    Chroma returns *distance* (lower is better). We convert to a similarity
    score in [0, 1] for a more intuitive threshold.
    """
    store = get_store()
    k = k or settings.top_k
    results = store.similarity_search_with_score(query, k=k)

    scored: list[tuple[Document, float]] = []
    for doc, distance in results:
        similarity = max(0.0, 1.0 - float(distance))
        if similarity >= settings.min_similarity:
            scored.append((doc, similarity))
    return scored


def annotate_freshness(doc: Document) -> str | None:
    """Return a human-readable warning if the source is older than STALE_DAYS."""
    ts = doc.metadata.get("last_modified_ts")
    if not ts:
        return None
    age = datetime.now() - datetime.fromtimestamp(ts)
    if age > timedelta(days=settings.stale_days):
        return (
            f"Bemærk: dokumentet '{doc.metadata.get('source_file')}' er fra "
            f"{doc.metadata.get('last_modified')} og kan være forældet."
        )
    return None
