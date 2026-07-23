"""RAG layer: changing facts live in a vector store, never in the weights.

The knowledge split is the framework's core anti-hallucination rule: LoRA
weights carry *behavior* (tone, format, reasoning style) while facts that
change — product catalogs, policies, uploaded PDFs — are retrieved at request
time from a :class:`VectorStore`.

The abstraction is deliberately tiny: an :class:`Embedder` callable and a
:class:`VectorStore` with ``add``/``search``. The built-in
:class:`InMemoryVectorStore` + :class:`HashEmbedder` pair is dependency-free
(character n-gram hashing + cosine similarity) so the whole pipeline runs in
tests and small deployments; production swaps in a real embedding model and a
persistent store behind the same two methods.
"""

from __future__ import annotations

import abc
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class Embedder(Protocol):
    def __call__(self, text: str) -> list[float]: ...


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")


@dataclass
class ScoredDocument:
    document: Document
    score: float


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n\n{para}".strip() if current else para
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - overlap:]
    if current:
        chunks.append(current)
    return chunks


class HashEmbedder:
    """Dependency-free embedding: hashed character n-gram counts.

    Not a semantic model — it captures lexical overlap, which is enough for
    tests and keyword-heavy corpora. Swap in a sentence-transformer or an
    embedding API for semantic retrieval; anything ``(str) -> list[float]``
    fits the seam.
    """

    def __init__(self, dim: int = 512, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def __call__(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        normalized = f" {' '.join(text.lower().split())} "
        for i in range(max(len(normalized) - self.ngram + 1, 1)):
            gram = normalized[i : i + self.ngram]
            vec[hash(gram) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # embedder outputs are normalized


class VectorStore(abc.ABC):
    """Minimal store contract; back with Chroma/Qdrant/pgvector as needed."""

    @abc.abstractmethod
    def add(self, documents: list[Document]) -> list[str]:
        """Index documents; returns their ids."""

    @abc.abstractmethod
    def search(self, query: str, k: int = 3) -> list[ScoredDocument]:
        """Top-k documents most similar to the query."""

    def add_texts(
        self, texts: list[str], metadata: dict[str, Any] | None = None
    ) -> list[str]:
        return self.add([Document(text=t, metadata=dict(metadata or {})) for t in texts])

    def add_document_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> list[str]:
        """Chunk one large document (e.g. an extracted PDF) and index it."""
        return self.add_texts(chunk_text(text, chunk_size, overlap), metadata)


class InMemoryVectorStore(VectorStore):
    """Cosine-similarity store over an injectable embedder."""

    def __init__(self, embedder: Embedder | Callable[[str], list[float]] | None = None):
        self.embedder = embedder or HashEmbedder()
        self._docs: dict[str, tuple[Document, list[float]]] = {}

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, documents: list[Document]) -> list[str]:
        for doc in documents:
            self._docs[doc.id] = (doc, self.embedder(doc.text))
        return [d.id for d in documents]

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)

    def search(self, query: str, k: int = 3) -> list[ScoredDocument]:
        query_vec = self.embedder(query)
        scored = [
            ScoredDocument(document=doc, score=_cosine(query_vec, vec))
            for doc, vec in self._docs.values()
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]


class Retriever:
    """Store + rendering policy: turns a query into a context block."""

    def __init__(self, store: VectorStore, k: int = 3, min_score: float = 0.0):
        self.store = store
        self.k = k
        self.min_score = min_score

    def retrieve(self, query: str) -> list[ScoredDocument]:
        return [s for s in self.store.search(query, self.k) if s.score >= self.min_score]

    def render_context(self, query: str) -> str:
        """Context block for the system prompt; empty string when nothing
        relevant is found (so callers can skip the section cleanly)."""
        hits = self.retrieve(query)
        if not hits:
            return ""
        sections = [f"[{i + 1}] {s.document.text}" for i, s in enumerate(hits)]
        return (
            "Answer using the following retrieved context; if it does not "
            "contain the answer, say you don't know rather than guessing.\n\n"
            + "\n\n".join(sections)
        )
