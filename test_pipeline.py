from src.rag.pipeline import ingest_text, query, reset_knowledge_base

# Reset first
reset_knowledge_base()

# Ingest sample text
sample = """
Artificial intelligence is transforming healthcare by enabling faster diagnosis
and more personalized treatment plans. Machine learning models can analyze medical
images with accuracy comparable to expert radiologists. Natural language processing
helps extract insights from clinical notes and medical literature.

RAG systems in healthcare reduce hallucinations by grounding responses in verified
medical literature. This is critical because incorrect medical information can
harm patients. Vector databases store embeddings of medical knowledge bases
enabling fast retrieval of relevant clinical guidelines.

The main challenges in healthcare AI include data privacy regulations like HIPAA,
bias in training data, and the need for model explainability. Clinicians need to
understand why an AI system made a recommendation before acting on it.
"""

print("Ingesting document...")
result = ingest_text(sample, source="healthcare_ai.txt")
print(f"Result: {result}\n")

# Test queries
questions = [
    "How does RAG help in healthcare?",
    "What are the main challenges of AI in healthcare?",
    "Summarize the key points about healthcare AI",
]

for q in questions:
    print(f"\nQ: {q}")
    result = query(q, verbose=True)
    print(f"Type: {result['query_type']} | "
          f"Strategy: {result['retrieval_strategy']} | "
          f"Confidence: {result['confidence']:.2f}")
    print(f"Answer: {result['answer'][:200]}...")