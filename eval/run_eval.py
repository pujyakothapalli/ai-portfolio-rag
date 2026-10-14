"""
RAGAS Evaluation: Adaptive RAG vs Naive RAG
Metrics: Faithfulness, Answer Relevancy, Context Precision

Usage:
    ANTHROPIC_API_KEY=sk-... python eval/run_eval.py
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings

from eval.test_dataset import TEST_CASES
from src.rag.pipeline import query as run_pipeline
from src.rag.retriever import HybridRetriever
from src.rag.generator import generate_answer


# ── LLM + Embeddings (no OpenAI) ────────────────────────────────────────────
def get_ragas_llm():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        anthropic_api_key=api_key,
    )
    return LangchainLLMWrapper(llm)


def get_ragas_embeddings():
    hf = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(hf)


# ── Naive RAG: always hybrid top-5, no classification ───────────────────────
_naive_retriever = None

def get_naive_retriever():
    global _naive_retriever
    if _naive_retriever is None:
        _naive_retriever = HybridRetriever()
    return _naive_retriever


def run_naive_rag(question: str) -> dict:
    retriever = get_naive_retriever()
    chunks = retriever.hybrid_retrieve(question, top_k=5)
    answer = generate_answer(question, chunks)
    return {
        "answer": answer,
        "contexts": [c["content"] for c in chunks],
    }


# ── Adaptive RAG ─────────────────────────────────────────────────────────────
def run_adaptive_rag(question: str) -> dict:
    result = run_pipeline(question)
    return {
        "answer": result["answer"],
        "contexts": [c["content"] for c in result.get("chunks", [])],
    }


# ── Build RAGAS Dataset ──────────────────────────────────────────────────────
def build_dataset(runner_fn, label: str) -> Dataset:
    print(f"\n{'='*60}")
    print(f"Running {label} on {len(TEST_CASES)} test cases...")
    print("=" * 60)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, tc in enumerate(TEST_CASES, 1):
        q = tc["question"]
        gt = tc["ground_truth"]
        qtype = tc["query_type"]
        print(f"[{i}/{len(TEST_CASES)}] [{qtype}] {q[:70]}...")
        try:
            result = runner_fn(q)
            ctx = result["contexts"]
            if not ctx:
                ctx = [""]
            questions.append(str(q))
            answers.append(str(result["answer"]))
            contexts.append([str(c) for c in ctx])
            ground_truths.append(str(gt))
            print(f"  ✓  chunks={len(ctx)}")
        except Exception as e:
            print(f"  ✗  ERROR: {e}")
            questions.append(str(q))
            answers.append("")
            contexts.append([""])
            ground_truths.append(str(gt))

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })


# ── Score ────────────────────────────────────────────────────────────────────
def score_dataset(dataset: Dataset, llm, embeddings) -> dict:
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    return {
        "faithfulness":      round(float(result["faithfulness"]), 4),
        "answer_relevancy":  round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
    }


# ── Print table ───────────────────────────────────────────────────────────────
def print_results(adaptive: dict, naive: dict):
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print("\n" + "=" * 65)
    print(f"{'Metric':<25} {'Adaptive RAG':>14} {'Naive RAG':>12} {'Delta':>10}")
    print("-" * 65)
    for m in metrics:
        a, n = adaptive[m], naive[m]
        delta = a - n
        print(f"{m:<25} {a:>14.4f} {n:>12.4f} {'+' if delta>=0 else ''}{delta:>9.4f}")
    print("=" * 65)
    avg_a = sum(adaptive.values()) / len(metrics)
    avg_n = sum(naive.values()) / len(metrics)
    d = avg_a - avg_n
    print(f"{'AVERAGE':<25} {avg_a:>14.4f} {avg_n:>12.4f} {'+' if d>=0 else ''}{d:>9.4f}")
    print("=" * 65)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    adaptive_dataset = build_dataset(run_adaptive_rag, "Adaptive RAG")
    naive_dataset    = build_dataset(run_naive_rag,    "Naive RAG")

    print("\nScoring Adaptive RAG with RAGAS...")
    adaptive_scores = score_dataset(adaptive_dataset, llm, embeddings)

    print("Scoring Naive RAG with RAGAS...")
    naive_scores = score_dataset(naive_dataset, llm, embeddings)

    print_results(adaptive_scores, naive_scores)

    output = {
        "adaptive_rag": adaptive_scores,
        "naive_rag":    naive_scores,
        "delta": {m: round(adaptive_scores[m] - naive_scores[m], 4) for m in adaptive_scores},
        "test_cases": len(TEST_CASES),
    }

    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()