from sentence_transformers import SentenceTransformer
import numpy as np

# powerful embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The patient has high blood pressure",
    "The patient suffers from hypertension", 
    "The stock market crashed today",
    "Investors lost money on Wall Street",
    "I love eating pizza",
    "Machine learning is a subset of AI",
]

# Generate embeddings
embeddings = model.encode(sentences)

print(f"Embedding shape: {embeddings[0].shape}")
print(f"Each sentence = vector of {embeddings[0].shape[0]} numbers\n")

# Compute cosine similarity between all pairs
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("=" * 60)
print("Similarity scores (1.0 = identical meaning, 0.0 = unrelated)")
print("=" * 60)

pairs = [
    (0, 1, "hypertension vs high blood pressure"),
    (2, 3, "stock market vs Wall Street investors"),
    (0, 2, "medical vs financial"),
    (4, 5, "pizza vs machine learning"),
    (0, 5, "medical vs ML"),
]

for i, j, label in pairs:
    score = cosine_similarity(embeddings[i], embeddings[j])
    bar = "█" * int(score * 20)
    print(f"{label}")
    print(f"  {score:.4f} {bar}\n")