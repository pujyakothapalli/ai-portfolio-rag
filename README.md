Markdown · README.md

# 🔍 Adaptive RAG System

A production-grade Retrieval-Augmented Generation pipeline that classifies query intent and routes each question to an optimized retrieval strategy — benchmarked against naive RAG using RAGAS.

[**🚀 Live Demo on Railway**](https://ai-portfolio-rag-production.up.railway.app)

---

## What It Does

Most RAG systems treat every question the same way: retrieve top-5 chunks, generate an answer. That works until you mix question types — a factual lookup needs tight, precise chunks; an analytical question needs broad context; a summarization task needs near-complete coverage.

This system solves that by classifying each query first, then routing it to the retrieval strategy that fits:

```
Query → Classifier → Router → [Dense | Hybrid] Retriever → Cross-Encoder Reranker → Generator → Answer
```

---

## Architecture

### 1. Query Classifier

Claude classifies each incoming question into one of four types using zero-shot prompting:

| Type | Description | Example |
| --- | --- | --- |
| `FACTUAL` | Specific lookup with a single clear answer | "What does LIME stand for?" |
| `ANALYTICAL` | Requires synthesis across multiple ideas | "What are the tradeoffs between X and Y?" |
| `SUMMARIZATION` | Broad coverage of a topic or document | "Summarize the main benefits of XAI" |
| `CONVERSATIONAL` | Context-aware, open-ended | "Can you give me an example?" |

### 2. Adaptive Router

Each query type maps to a different retrieval configuration:

| Query Type | Strategy | Retrieve (k) | Rerank (n) |
| --- | --- | --- | --- |
| FACTUAL | Dense | 6 | 3 |
| ANALYTICAL | Hybrid | 10 | 5 |
| SUMMARIZATION | Hybrid | 10 | 5 |
| CONVERSATIONAL | Dense | 6 | 3 |

### 3. Hybrid Retrieval

- **BM25** (sparse): keyword-exact matching via `rank_bm25`
- **Dense** (semantic): vector similarity via ChromaDB + `all-MiniLM-L6-v2` embeddings
- **Hybrid**: BM25 + dense scores fused with Reciprocal Rank Fusion

### 4. Cross-Encoder Reranking

Retrieved candidates are re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers). This is a retrieve-wide-then-rerank pattern — the system over-retrieves (k=6–10) then collapses to the highest-scoring n chunks. This is the biggest driver of the context precision improvement.

### 5. Self-RAG Confidence Loop

After generation, Claude scores its own answer confidence (0–100). If below 70, it retries with a broader retrieval strategy before returning.

### 6. Generation

Claude generates the final answer with retrieved + reranked context, citing sources per chunk.

---

## Evaluation Results (RAGAS)

Benchmarked on **10 test cases** grounded in *"Explainable AI in Healthcare"* (Pawar, O'Shea, Rea, O'Reilly — Cork Institute of Technology) using **RAGAS v0.1.21** with Claude as judge and `all-MiniLM-L6-v2` for embeddings (no OpenAI dependency).

### Metrics Explained

- **Faithfulness**: Does the answer stick to what the retrieved context actually says? (no hallucination)
- **Answer Relevancy**: Does the answer address what was actually asked?
- **Context Precision**: Of the retrieved chunks, how many were actually useful?

### Results

| Metric | Adaptive RAG | Naive RAG | Delta |
| --- | --- | --- | --- |
| Faithfulness | 0.9394 | 0.8594 | **+8.0%** |
| Answer Relevancy | 0.8225 | 0.8200 | **+0.3%** |
| Context Precision | 0.6644 | 0.3917 | **+27.3%** |
| **Average** | **0.8088** | **0.6904** | **+11.8%** |

**Naive RAG baseline**: always hybrid top-5, no query classification, no reranking.

The headline number is context precision (+27.3 pp). Reranking dramatically reduces retrieved noise — the system surfaces the right chunks rather than a mix of relevant and tangential ones. Faithfulness at 0.94 means the answers are closely grounded in source material.

---

## Tech Stack

| Layer | Tool | Purpose |
| --- | --- | --- |
| LLM | Anthropic Claude (`claude-haiku-4-5`) | Classification, generation, Self-RAG scoring |
| Orchestration | LangChain | Pipeline wiring |
| Vector Store | ChromaDB | Dense embeddings storage and retrieval |
| Sparse Retrieval | rank-bm25 | Keyword-based BM25 retrieval |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Document and query embeddings |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Post-retrieval candidate reranking |
| Evaluation | RAGAS v0.1.21 | Faithfulness, Answer Relevancy, Context Precision |
| UI | Streamlit | Frontend |
| Hosting | Hugging Face Spaces | Live deployment |

---

## Hosting

The app is deployed on **Railway** as a containerized Streamlit service. The `ANTHROPIC_API_KEY` is set as a Railway environment variable. The ChromaDB vector store is ephemeral per session — documents you upload are indexed in-memory and persist for the session. For a persistent deployment, swap ChromaDB for a hosted vector DB (Pinecone, Weaviate, etc.).

The RAGAS evaluation was run locally using Claude as the judge LLM — no OpenAI API key required.

---

## How to Use (Live Demo)

1. Open the [live demo](https://huggingface.co/spaces/pujyakothapalli/adaptive-rag)
2. Upload a PDF or paste text in the sidebar
3. Ask any question in the chat
4. Each response includes:
   - The answer with cited sources
   - Query type detected (FACTUAL / ANALYTICAL / SUMMARIZATION / CONVERSATIONAL)
   - Retrieval strategy used (dense / hybrid)
   - Number of chunks retrieved and reranked
   - Confidence score (Self-RAG)

---

## Run Locally

### Prerequisites

- Python 3.10+
- Anthropic API key

### Setup

```bash
git clone https://github.com/pujyakothapalli/adaptive-rag
cd adaptive-rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Run the Evaluation

```bash
python eval/run_eval.py
```

Results are written to `eval/results.json`. Requires `ANTHROPIC_API_KEY` — uses Claude as the RAGAS judge (no OpenAI key needed).

---

## Project Structure

```
adaptive-rag/
├── app.py                    # Streamlit UI
├── src/
│   └── rag/
│       ├── pipeline.py       # Main query() entrypoint — routing + reranking
│       ├── classifier.py     # Claude-based query type classifier
│       ├── retriever.py      # HybridRetriever + CrossEncoder rerank()
│       ├── generator.py      # Claude generation + Self-RAG scoring
│       └── ingestion.py      # PDF/text chunking + embedding + indexing
├── eval/
│   ├── run_eval.py           # RAGAS evaluation script
│   ├── test_dataset.py       # 10 test cases grounded in the source paper
│   └── results.json          # Latest benchmark output
├── requirements.txt
└── .env.example
```

---

## Key Design Decisions

**Why classify queries instead of always using hybrid?** Hybrid retrieval is great for broad questions but can dilute precision on factual lookups. Dense-only on factual queries gets tighter, more relevant chunks.

**Why rerank instead of just retrieving a smaller k?** A cross-encoder scores query–chunk pairs jointly (not independently), catching relevance signals that embedding similarity misses. Over-retrieving then reranking consistently outperforms retrieving a small k directly.

**Why no OpenAI dependency?** HuggingFace embeddings (`all-MiniLM-L6-v2`) replace OpenAI embeddings everywhere — in the pipeline and in RAGAS evaluation. This keeps the project fully runnable with just an Anthropic key.

---

## Part of a 4-Project AI/ML Portfolio

This is Project 1 of a series building toward production-grade AI systems:

| # | Project | Status |
| --- | --- | --- |
| 1 | **Adaptive RAG System** (this repo) | ✅ Complete |
| 2 | Autonomous Research Agent (LangGraph + Tavily) | 🔄 In progress |
| 3 | TBD | — |
| 4 | TBD | — |

---

*Built by [Pujya Kothapalli](https://github.com/pujyakothapalli) — MS Data Science, University of Maryland*