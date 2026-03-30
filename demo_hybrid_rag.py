"""Hybrid RAG demo: keyword + vector retrieval, RRF fusion, and re-ranking.

Embedding models  : Voyage AI voyage-3    (embedding)
                                        Voyage AI rerank-2.5  (re-ranking cross-encoder)

Set VOYAGE_API_KEY in your environment to enable real Voyage AI calls.
Without the key both stages fall back to token-overlap similarity / scoring,
which is sufficient to demonstrate the pipeline flow.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Iterable

try:
    import voyageai

    _VOYAGE_AVAILABLE = True
except ImportError:
    _VOYAGE_AVAILABLE = False

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = "voyage-3"
RERANK_MODEL = "rerank-2.5"
EMBEDDING_SIMILARITY_THRESHOLD = 0.5


@dataclass
class Document:
    page_content: str
    metadata: dict[str, str] = field(default_factory=dict)
    embedding: list[float] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class KeywordRetriever:
    """BM25-style token-overlap retrieval (demo approximation)."""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def invoke(self, query: str) -> list[Document]:
        query_terms = normalize(query)
        ranked = sorted(
            self.documents,
            key=lambda doc: keyword_score(query_terms, normalize(doc.page_content)),
            reverse=True,
        )
        return [
            doc for doc in ranked
            if keyword_score(query_terms, normalize(doc.page_content)) > 0
        ]


class VectorRetriever:
    """Semantic retrieval using Voyage AI embeddings (voyage-3).

    Falls back to token-overlap similarity when VOYAGE_API_KEY is not set
    or the voyageai package is not installed.
    """

    def __init__(self, documents: list[Document], api_key: str | None = None) -> None:
        self.documents = documents
        self._client = None

        use_voyage = _VOYAGE_AVAILABLE and api_key is not None
        if use_voyage:
            self._client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
            texts = [doc.page_content for doc in documents]
            result = self._client.embed(texts, model=VOYAGE_MODEL, input_type="document")
            for doc, emb in zip(documents, result.embeddings):
                doc.embedding = emb
        else:
            mode = "token-overlap fallback (no VOYAGE_API_KEY)"
            print(f"[VectorRetriever] Using {mode}")

    def invoke(self, query: str) -> list[Document]:
        if self._client is not None:
            return self._embed_and_rank(query)
        return self._token_rank(query)

    def _embed_and_rank(self, query: str) -> list[Document]:
        result = self._client.embed([query], model=VOYAGE_MODEL, input_type="query")  # type: ignore[union-attr]
        query_vec = result.embeddings[0]
        ranked = sorted(
            self.documents,
            key=lambda doc: cosine_similarity(query_vec, doc.embedding or []),
            reverse=True,
        )
        return [
            doc for doc in ranked
            if cosine_similarity(query_vec, doc.embedding or []) >= EMBEDDING_SIMILARITY_THRESHOLD
        ]

    def _token_rank(self, query: str) -> list[Document]:
        query_terms = expand_semantics(normalize(query))
        ranked = sorted(
            self.documents,
            key=lambda doc: semantic_score(query_terms, expand_semantics(normalize(doc.page_content))),
            reverse=True,
        )
        return [
            doc for doc in ranked
            if semantic_score(query_terms, expand_semantics(normalize(doc.page_content))) > 0
        ]


class VoyageReranker:
    """Cross-encoder re-ranking using Voyage AI (rerank-2.5).

    Falls back to token-overlap cross-encoder when VOYAGE_API_KEY is not set
    or the voyageai package is not installed.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = None
        if _VOYAGE_AVAILABLE and api_key is not None:
            self._client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        else:
            print("[VoyageReranker] Using token-overlap fallback (no VOYAGE_API_KEY)")

    def compress_documents(
        self, documents: list[Document], query: str, top_k: int = 3
    ) -> list[Document]:
        if not documents:
            return []
        if self._client is not None:
            return self._rerank(documents, query, top_k)
        return self._token_rerank(documents, query)[:top_k]

    def _rerank(self, documents: list[Document], query: str, top_k: int) -> list[Document]:
        texts = [doc.page_content for doc in documents]
        result = self._client.rerank(  # type: ignore[union-attr]
            query, texts, model=RERANK_MODEL, top_k=top_k
        )
        return [documents[r.index] for r in result.results]

    def _token_rerank(self, documents: list[Document], query: str) -> list[Document]:
        query_terms = expand_semantics(normalize(query))
        return sorted(
            documents,
            key=lambda doc: rerank_score(query_terms, expand_semantics(normalize(doc.page_content))),
            reverse=True,
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> set[str]:
    cleaned = text.lower()
    for char in ",.?()[]:-_/":
        cleaned = cleaned.replace(char, " ")
    return {token for token in cleaned.split() if token}


def expand_semantics(tokens: Iterable[str]) -> set[str]:
    synonyms: dict[str, set[str]] = {
        "combine": {"combine", "merge", "fusion"},
        "fusion": {"fusion", "combine", "merge"},
        "algorithm": {"algorithm", "method", "approach"},
        "keyword": {"keyword", "bm25", "lexical", "full", "text"},
        "vector": {"vector", "semantic", "embedding", "dense"},
        "scores": {"scores", "ranking", "rank"},
        "comparable": {"comparable", "calibrated", "aligned"},
        "retrieval": {"retrieval", "search", "retrieve"},
        "hallucination": {"hallucination", "incorrect", "wrong", "error"},
        "context": {"context", "document", "chunk", "passage"},
    }
    expanded: set[str] = set()
    for token in tokens:
        expanded.update(synonyms.get(token, {token}))
    return expanded


def keyword_score(query_terms: set[str], doc_terms: set[str]) -> int:
    return len(query_terms & doc_terms)


def semantic_score(query_terms: set[str], doc_terms: set[str]) -> int:
    return len(query_terms & doc_terms)


def rerank_score(query_terms: set[str], doc_terms: set[str]) -> tuple[int, int]:
    exact_hits = len(query_terms & doc_terms)
    token_count = len(doc_terms)
    return exact_hits, -token_count


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.metadata["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_context(
    query: str,
    keyword_retriever: KeywordRetriever,
    vector_retriever: VectorRetriever,
    reranker: VoyageReranker,
) -> list[Document]:
    keyword_docs = keyword_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)

    fused_scores = reciprocal_rank_fusion([keyword_docs, vector_docs])
    doc_pool = {doc.metadata["id"]: doc for doc in keyword_docs + vector_docs}

    top_docs = sorted(
        doc_pool.values(),
        key=lambda doc: fused_scores.get(doc.metadata["id"], 0.0),
        reverse=True,
    )[:5]

    return reranker.compress_documents(top_docs, query, top_k=3)


# ---------------------------------------------------------------------------
# Dataset (10 documents)
# ---------------------------------------------------------------------------

def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Reciprocal Rank Fusion combines ranked lists without comparing raw scores.",
            metadata={"id": "doc-01-rrf"},
        ),
        Document(
            page_content="Relative Score Fusion works best when scores are normalized and calibrated across retrievers.",
            metadata={"id": "doc-02-rsf"},
        ),
        Document(
            page_content="BM25 is a strong keyword retrieval method for exact terms, product codes, and acronyms.",
            metadata={"id": "doc-03-bm25"},
        ),
        Document(
            page_content="Dense vector retrieval finds semantically similar documents even when wording differs from the query.",
            metadata={"id": "doc-04-vector"},
        ),
        Document(
            page_content="Cross-encoder re-ranking scores each query-document pair directly, improving final context quality.",
            metadata={"id": "doc-05-rerank"},
        ),
        Document(
            page_content="Hybrid search combines keyword and semantic retrieval to improve both recall and precision.",
            metadata={"id": "doc-06-hybrid"},
        ),
        Document(
            page_content="RAG pipelines reduce hallucinations by grounding LLM generation in retrieved context.",
            metadata={"id": "doc-07-rag"},
        ),
        Document(
            page_content="Chunking strategy affects retrieval quality: smaller chunks increase precision, larger chunks preserve context.",
            metadata={"id": "doc-08-chunking"},
        ),
        Document(
            page_content="Voyage AI provides embedding models optimized for retrieval tasks, including voyage-3 and voyage-3-lite.",
            metadata={"id": "doc-09-voyage"},
        ),
        Document(
            page_content="LangChain retrievers expose a common invoke interface that works with keyword, vector, and hybrid backends.",
            metadata={"id": "doc-10-langchain"},
        ),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    query = "Which fusion algorithm is better for combining keyword and vector search when scores are not comparable?"

    documents = sample_documents()
    keyword_retriever = KeywordRetriever(documents)
    vector_retriever = VectorRetriever(documents, api_key=VOYAGE_API_KEY)
    reranker = VoyageReranker(api_key=VOYAGE_API_KEY)

    final_context = build_context(query, keyword_retriever, vector_retriever, reranker)

    print("Embedding model : Voyage AI (voyage-3)" if VOYAGE_API_KEY else "Embedding model : token-overlap fallback")
    print("Reranker model  : Voyage AI (rerank-2.5)" if VOYAGE_API_KEY else "Reranker model  : token-overlap fallback")
    print(f"Dataset size    : {len(documents)} documents")
    print(f"\nQuery: {query}")
    print("\nTop context after hybrid retrieval + RRF fusion + re-ranking:")

    for index, document in enumerate(final_context, start=1):
        print(f"  {index}. [{document.metadata['id']}] {document.page_content}")


if __name__ == "__main__":
    main()