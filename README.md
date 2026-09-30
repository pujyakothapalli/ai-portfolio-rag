---
title: Adaptive RAG System
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.64.0
app_file: app.py
pinned: false
---

# Adaptive RAG System

Production-grade Retrieval-Augmented Generation with:
- **Adaptive query routing** — classifies intent, picks optimal retrieval strategy
- **Hybrid search** — BM25 sparse + dense vector retrieval combined
- **Self-RAG loop** — scores confidence, retries if below 70%
- **RAGAS benchmarked** — measurable improvement over naive RAG

## Stack
Python · ChromaDB · Sentence Transformers · Anthropic Claude · Streamlit

## How to use
1. Upload any PDF or paste text in the sidebar
2. Ask questions in the chat
3. See confidence scores, retrieval strategy, and sources for every answer