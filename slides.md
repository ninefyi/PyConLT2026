---
marp: true
theme: default
math: katex
paginate: true
size: 16:9
title: "Beyond Basic RAG: Boosting Accuracy with Hybrid Search and Fusion Algorithms"
description: PyCon LT 2026 talk deck
---

<style>
section {
  font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
  color: #10233f;
  background:
    radial-gradient(circle at 85% 10%, #d9f1ff 0%, #d9f1ff 14%, transparent 15%),
    radial-gradient(circle at 12% 88%, #ffe9d6 0%, #ffe9d6 11%, transparent 12%),
    linear-gradient(135deg, #f7fbff 0%, #eef7ff 45%, #f9fcff 100%);
}

h1,
h2,
h3 {
  color: #0e2a66;
  font-weight: 700;
}

strong {
  color: #1f5fbf;
}

code,
pre,
kbd,
samp {
  font-family: "JetBrains Mono", "Consolas", "Courier New", monospace;
}

pre {
  background: #f3f8ff;
  border: 1px solid #c8dcff;
  border-radius: 10px;
  padding: 12px;
  font-size: 0.73em;
  line-height: 1.4;
}

table {
  border-collapse: collapse;
  font-size: 0.78em;
  margin: auto;
}

table th {
  background: #0e2a66;
  color: #ffffff;
}

table th,
table td {
  border: 1px solid #c6d8f3;
  padding: 4px 8px;
}

blockquote {
  border-left: 6px solid #f59f3a;
  background: #fff5ea;
  padding: 8px 14px;
}
</style>

# Beyond Basic RAG: Boosting Accuracy with Hybrid Search and Fusion Algorithms

PyCon LT 2026

Piti Champeethong

10-Apr-2026

---

# Agenda

1. Evolution of the recommender system
2. Why vector-only retrieval breaks in real RAG
3. Introduction to Hybrid Search
4. Introduction to the Fusion algorithm
5. Proposed RAG pipeline
6. Key takeaway
7. Q&A

---

# Why This Talk

- RAG quality depends on retrieval quality.
- Vector search is useful, but not enough on its own.
- Exact keywords, product codes, acronyms, and domain terms often matter.
- Better retrieval means better context.
- Better context means fewer hallucinations.

---

# The Evolution of Recommender Systems

| Era | Main idea | Strength | Limitation |
| --- | --- | --- | --- |
| Rule-based | Hard-coded conditions | Easy to explain | Brittle |
| Collaborative filtering | Learn from user behavior | Personalization | Cold start |
| Content-based | Match item features | Works with metadata | Limited understanding |
| Deep retrieval | Learn dense representations | Captures meaning | Misses exact terms |

---

# From Recommendation to Retrieval

- Recommender systems rank useful items for a user.
- Search systems rank useful documents for a query.
- Modern RAG uses retrieval as the first ranking step.
- If the wrong documents are retrieved, generation starts from weak context.
- In practice, retrieval is the real foundation of RAG quality.

---

# What Vector Search Does Well

- Finds semantically similar text.
- Handles paraphrases and related concepts.
- Works well when the query and document use different wording.
- Helps when user questions are natural language and indirect.

Example:

- Query: `How do I reset a forgotten password?`
- Relevant document: `Account recovery procedure`

---

# Where Vector Search Fails

- Exact keywords may be diluted in dense embeddings.
- Acronyms can be ambiguous.
- Product codes and IDs often need exact matching.
- Domain-specific terms may be rare in general embeddings.

Examples:

- `RRF` should not be confused with unrelated text.
- `SKU-8472-B` must match exactly.
- `BM25` and `cross-encoder` should not disappear in semantic noise.

---

# Why Vector-Only RAG Misses Useful Context

Query:

> Which fusion algorithm is better for combining keyword search and vector
> search when raw scores are not directly comparable?

Possible failure:

- Vector search retrieves documents about ranking in general.
- It misses the document that explicitly explains `Reciprocal Rank Fusion`.
- The generator answers with partial or incorrect reasoning.

---

# Introduction to Hybrid Search

Hybrid Search combines:

- Keyword or full-text retrieval
- Semantic or vector retrieval

Goal:

- Keep exact term matching
- Keep semantic understanding
- Use both signals to retrieve stronger candidates

---

# Keyword vs. Semantic Retrieval

| Retrieval type | Best at | Weak at |
| --- | --- | --- |
| Keyword search | Exact tokens, acronyms, IDs, filters | Synonyms and paraphrases |
| Vector search | Meaning, paraphrases, related ideas | Exact keywords and rare terms |
| Hybrid search | Balancing both worlds | Needs a fusion strategy |

---

# Why Hybrid Search Matters

- Real business data mixes plain language and exact terminology.
- Users ask in natural language, but documents may contain strict keywords.
- Hybrid retrieval improves recall without giving up precision.
- It is usually a safer default than vector-only retrieval.

Short version:

> Use keyword search for exactness and vector search for meaning.

---

# A Simple Hybrid Retrieval Pattern

```python
from langchain_core.documents import Document


def hybrid_candidates(query, keyword_retriever, vector_retriever):
    keyword_docs = keyword_retriever.invoke(query)
    vector_docs = vector_retriever.invoke(query)
    return keyword_docs, vector_docs
```

- Run two retrieval strategies.
- Merge the results later.
- Keep the code easy to explain.

---

# Introduction to the Fusion Algorithm

Hybrid search gives two ranked lists.

Now we need one combined ranking.

Two common choices:

- Reciprocal Rank Fusion, or `RRF`
- Relative Score Fusion, or `RSF`

The choice matters because keyword and vector scores behave differently.

---

# Reciprocal Rank Fusion, or RRF

**Core idea:**

- Ignore raw scores.
- Use only the rank position from each retrieval list.
- Reward documents that appear near the top in multiple lists.

**Formula:**

$$
RRF(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
$$

---

# Why RRF Works Well

- **Simple** — easy to understand and implement
- **Stable** — rank positions are less noisy than scores
- **Robust** — works when score scales are different

This makes it the practical default for hybrid search.

---

# RRF: Worked Example

Three documents, two retrievers, `k = 60`:

| Doc | KW rank | V rank | KW score | V score | **RRF** |
| --- | --- | --- | --- | --- | --- |
| A | 1 | 2 | 0.0164 | 0.0161 | **0.0325** |
| B | 3 | 1 | 0.0159 | 0.0164 | **0.0323** |
| C | 2 | 3 | 0.0161 | 0.0159 | **0.0320** |

**Final ranking:** Doc A → Doc B → Doc C

**Key insight:** Doc A wins because it ranked near the top in both lists.

---

# Relative Score Fusion, or RSF

Idea:

- Normalize scores from each retriever.
- Combine the normalized values.

Typical pattern:

$$
RSF(d) = w_1 \cdot s_1(d) + w_2 \cdot s_2(d)
$$

Why it can help:

- You can tune weights.
- You can favor one retriever more than another.

Risk: Raw scores from different systems are often not comparable.

---

# RSF: Worked Example

Same three documents, normalized to 0–1, weights `w1 = 0.5, w2 = 0.5`:

| Doc | KW score | V score | **0.5×KW + 0.5×V** |
| --- | --- | --- | --- |
| A | 0.90 | 0.50 | **0.70** |
| B | 0.30 | 0.80 | **0.55** |
| C | 0.60 | 0.40 | **0.50** |

**Final ranking:** Doc A → Doc B → Doc C

**⚠️ Warning:** RSF breaks without normalization—scores from different systems often have different scales.

---

# RRF vs. RSF

| Method | Best when | Trade-off |
| --- | --- | --- |
| RRF | Retrieval scores come from different scales | Less control over score weighting |
| RSF | Scores are normalized well and weights matter | Easier to misuse |

Practical advice:

- Start with `RRF`.
- Move to `RSF` only when you trust score calibration.

---

# Core Fusion Snippet

```python
def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}

    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.metadata["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return scores
```

- Short enough for beginners to follow.
- Works even when retrievers produce unrelated score ranges.

---

# MongoDB 8.0+: Native RRF with `$rankFusion`

MongoDB 8.0 ships `$rankFusion` as a built-in aggregation stage.

No custom fusion code needed — the database handles it natively.

What it does:

- Runs multiple input pipelines in parallel.
- Combines and de-duplicates results.
- Scores each document using the RRF formula.
- Supports optional pipeline **weights** for fine-tuning.

---

# MongoDB `$rankFusion`: Supported Types

**Input pipeline types:**

- `$vectorSearch` — dense semantic retrieval
- `$search` — full-text keyword retrieval (Atlas Search)
- `$geoNear`, `$match`, `$sort` — other filters

**Learn more:**

[MongoDB rankFusion docs](https://www.mongodb.com/docs/manual/reference/operator/aggregation/rankFusion/)

---

# MongoDB `$rankFusion`: Part 1

```js
db.movies.aggregate([
  {
    $rankFusion: {
      input: {
        pipelines: {
          vectorPipeline: [
            {
              $vectorSearch: {
                index: "vector_index",
                path: "embedding",
                queryVector: [0.1, 0.3, ...],
                limit: 20
              }
            }
          ],
          keywordPipeline: [
            {
              $search: {
                text: { query: "my query", path: "plot" }
              }
            },
            { $limit: 20 }
          ]
        }
      }
    }
  }
])
```

---

# MongoDB `$rankFusion`: Part 2

**Optional fine-tuning:**

```js
combination: {
  weights: { vectorPipeline: 2, keywordPipeline: 1 },
  scoreDetails: true  // exposes rank and score per pipeline
}
```

**After fusion:**

```js
{ $limit: 10 }  // keep top 10 results
```

- Native RRF: no custom fusion code needed.
- Weights let you bias one retriever over another.

---

# Proposed RAG Pipeline

1. Receive a user query.
2. Run keyword retrieval.
3. Run vector retrieval.
4. Fuse both result lists.
5. Keep the top candidates.
6. Re-rank with a cross-encoder.
7. Send the best context to the LLM.
8. Generate the final answer.

---

# Pipeline View

```text
User Query
   |
   +--> Keyword Retriever -----+
   |                           |
   +--> Vector Retriever ------+--> Fusion --> Top N --> Re-ranker --> LLM
```

Key point:

- Fusion improves recall.
- Re-ranking improves precision.

---

# Why Add Re-ranking

- Retrieval gives a good candidate set.
- Re-ranking gives a better final order.
- A cross-encoder reads query and document together.
- It is slower than retrieval, so use it on a small top-N set.

This is a useful pattern:

- Fast retrieval first
- Expensive precision step second

---

# Re-ranking with a Cross-Encoder

What it does:

- Score each query-document pair directly.
- Understand token interaction better than vector distance alone.
- Promote the most relevant chunks before generation.

What it improves:

- **Relevance** — direct query-document signal
- **Context quality** — best chunks selected first
- **Answer grounding** — stronger evidence for LLM

---

# Re-ranking: How the Score Works

A cross-encoder reads **query + document together**, then outputs one relevance score:

$$
score = CrossEncoder(query,\; document)
$$

- Higher score → more relevant
- Unlike vector search, it sees **both texts at once**
- Captures phrase match, context, and intent together

---

# Re-ranking: Worked Example

**Query:** `Which algorithm merges ranked lists without comparing raw scores?`

| # | Document snippet | Score |
| --- | --- | --- |
| 1 | *"RRF combines ranked lists without raw scores"* | **9.4** |
| 2 | *"BM25 finds exact keyword matches"* | 4.1 |
| 3 | *"RSF uses normalized scores with weights"* | 3.8 |

**Result:** Doc 1 is promoted to the top.

**Key insight:** Re-ranking surfaces the best chunk that retrieval alone might bury.

---

# Using Voyage AI for Re-ranking

```python
import voyageai

vo = voyageai.Client()
result = vo.rerank(
  query="Which algorithm for incomparable scores?",
  documents=["doc1", "doc2", "doc3"],
  model="rerank-2.5",
  top_k=3,
)

for r in result.results:
  print(r.relevance_score, r.document[:40])
```

- Production-grade cross-encoder
- Simple, clean API
- Uses `VOYAGE_API_KEY` from environment

---

# Complete RAG Pipeline Example

```python
def build_context(query, keyword_retriever,
                  vector_retriever, reranker):
    # 1. Retrieve from both sources
    kw = keyword_retriever.invoke(query)
    v_docs = vector_retriever.invoke(query)

    # 2. Fuse results
    scores = reciprocal_rank_fusion([kw, v_docs])
    doc_pool = {d.metadata["id"]: d
                for d in kw + v_docs}

    # 3. Top 8 by fusion score
    top_docs = sorted(
      list(doc_pool.values()),
      key=lambda d: scores.get(d.metadata["id"], 0),
      reverse=True
    )[:8]

    # 4. Re-rank to get best 4
    return reranker.compress_documents(
      top_docs, query, top_k=4
    )
```

---

# What Each Step Does

| Step | Purpose |
| --- | --- |
| Keyword retrieval | Protect exact terms, codes, acronyms |
| Vector retrieval | Capture semantic meaning |
| Fusion | Combine both signals |
| Re-ranking | Top chunks first |
| Generation | Quality answer |

---

# Example Failure Before and After

Without hybrid search:

- Query asks for `RRF`.
- Retrieved chunks discuss ranking, but not the exact algorithm.
- LLM guesses and mixes up concepts.

With hybrid search and re-ranking:

- Exact `RRF` document is retrieved.
- Semantic support documents are also included.
- Re-ranker pushes the most relevant chunk to the top.
- Final answer is more precise and grounded.

---

# When This Pattern Helps Most

- Enterprise knowledge bases
- Technical documentation
- Internal support bots
- Product catalogs with codes and names
- Any domain with acronyms and exact terms

If your data contains both natural language and strict identifiers,
hybrid retrieval is usually worth it.

---

# Key Takeaway

- Vector search is strong, but not complete.
- Keyword search protects exact terms.
- Hybrid retrieval combines exactness and meaning.
- `RRF` is a practical default fusion strategy.
- Re-ranking improves the final context seen by the LLM.
- Better context reduces hallucinations.

---

# Q&A

Thank you.
