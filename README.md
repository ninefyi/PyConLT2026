# Beyond Basic RAG: Hybrid Search with Fusion and Re-ranking

A concise demo project for improving Retrieval-Augmented Generation (RAG) quality using hybrid retrieval, Reciprocal Rank Fusion (RRF), and re-ranking.

## Overview

This repository demonstrates a practical RAG pipeline:

- Keyword retrieval for exact terms
- Vector retrieval for semantic similarity
- RRF to combine ranked lists
- Re-ranking for better final context

The demo runs with Voyage AI models when `VOYAGE_API_KEY` is set, and falls back to local token-overlap scoring when it is not.

## Project Files

- `demo_hybrid_rag.py`: End-to-end hybrid RAG demo
- `requirements.txt`: Python dependencies
- `slides.md`: Marp presentation deck
- `.devcontainer/`: Containerized dev setup

## Prerequisites

- Python 3.10+
- pip
- Optional: Voyage AI API key

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Optional: set API key for real embedding and reranking:

```bash
export VOYAGE_API_KEY="your-api-key"
```

1. Run the demo:

```bash
python demo_hybrid_rag.py
```

## Expected Output

The script prints:

- Active embedding mode (Voyage or fallback)
- Active reranker mode (Voyage or fallback)
- Dataset size and query
- Top ranked context documents after hybrid retrieval + RRF + re-ranking

## Slides

`slides.md` is a Marp deck for the PyCon LT 2026 talk.

To export slides:

```bash
marp slides.md
```

## Dev Container

The dev container includes Python, markdown linting, and Marp CLI.

See:

- `.devcontainer/devcontainer.json`
- `.devcontainer/Dockerfile`

## Notes

- RRF is a strong default when retriever score scales are not directly comparable.
- Re-ranking is most effective after retrieval narrows candidates to a small top-N set.
