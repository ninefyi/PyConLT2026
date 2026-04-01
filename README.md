# Beyond Basic RAG: Hybrid Search, Fusion, and Re-ranking

This repository contains PyCon LT 2026 demo materials for building higher-accuracy Retrieval-Augmented Generation (RAG) with MongoDB Atlas, Voyage AI, and LangChain.

## Overview

The project demonstrates two practical hybrid RAG patterns:

1. Native MongoDB Atlas `$rankFusion` + Voyage AI re-ranking.
1. Manual hybrid retrieval (`$search` + LangChain vector retrieval) + Python RRF + Voyage AI re-ranking.

Both notebook flows include at least 10 examples and are designed to run safely in GitHub Codespaces.

## Project Structure

- `01_hybrid_rag_rankfusion.ipynb`: Notebook 1 using Atlas `$rankFusion`.
- `02_hybrid_rag_manual_rrf.ipynb`: Notebook 2 using manual RRF.
- `slides.md`: Marp slide deck used for the talk.
- `requirements.txt`: Python package dependencies.
- `.devcontainer/devcontainer.json`: Dev container configuration.
- `.devcontainer/Dockerfile`: Dev container base image and tooling.

## Prerequisites

- Python 3.10+ (virtual environment recommended)
- MongoDB Atlas cluster
- Voyage AI API key

## Environment Variables

Set these before running the notebooks:

```bash
export VOYAGE_API_KEY="your-voyage-api-key"
export MONGODB_URI="your-mongodb-atlas-uri"
```

If either value is missing, notebook cells skip external operations gracefully so execution does not fail.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Notebooks

### Notebook 1: Atlas `$rankFusion`

File: `01_hybrid_rag_rankfusion.ipynb`

Flow:

1. Create LangChain documents.
1. Store vectors with `MongoDBAtlasVectorSearch`.
1. Run native Atlas `$rankFusion` for keyword + vector fusion.
1. Re-rank top candidates with Voyage AI `rerank-2.5`.

### Notebook 2: Manual RRF

File: `02_hybrid_rag_manual_rrf.ipynb`

Flow:

1. Keyword retrieval via Atlas `$search`.
1. Vector retrieval via LangChain `MongoDBAtlasVectorSearch.similarity_search_with_score`.
1. Manual Reciprocal Rank Fusion (RRF) in Python.
1. Re-rank final candidates with Voyage AI `rerank-2.5`.

## Slides

The presentation deck is in `slides.md`.

Export with Marp:

```bash
marp slides.md
```

## Key Dependencies

- `voyageai`
- `langchain-core`
- `langchain-mongodb`
- `pymongo`
- `python-dotenv`
- `ipykernel`

## Notes

- Vector store integration follows LangChain MongoDB Atlas VectorStore patterns.
- Notebook cells are organized for step-by-step demo delivery.
- For Atlas search/index setup, ensure your cluster supports the used operators and index definitions.
