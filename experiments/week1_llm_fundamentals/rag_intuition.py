from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Simulate a tiny document store
documents = [
    "Python is a high-level programming language known for simplicity",
    "Neural networks are inspired by the human brain structure",
    "Hypertension is a condition where blood pressure is consistently high",
    "The transformer architecture uses self-attention mechanisms",
    "Diabetes is a metabolic disease causing high blood sugar",
    "Gradient descent optimizes model weights by minimizing loss",
    "REST APIs allow communication between software applications",
    "BERT is a bidirectional transformer model pretrained on large text",
]

# User queries
queries = [
    "how do transformers work in deep learning?",
    "what causes high blood pressure?",
    "how does a neural network learn?",
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

doc_embeddings = model.encode(documents)

print("=" * 60)
for query in queries:
    query_embedding = model.encode(query)
    
    # Score all documents
    scores = [(cosine_similarity(query_embedding, doc_emb), doc) 
              for doc_emb, doc in zip(doc_embeddings, documents)]
    scores.sort(reverse=True)
    
    print(f"\nQuery: '{query}'")
    print("Top 3 matches:")
    for score, doc in scores[:3]:
        print(f"  {score:.4f} — {doc}")
print("=" * 60)