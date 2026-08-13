import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

# ── Setup ──────────────────────────────────────────────
# Initialize ChromaDB — stores data in memory for now
client = chromadb.Client()

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.\n")

# ── EXPERIMENT 1: Basic ChromaDB operations ────────────
print("=" * 60)
print("EXPERIMENT 1: ChromaDB basics")
print("=" * 60)

# Create a collection (like a table in a database)
collection = client.create_collection(
    name="ai_concepts",
    metadata={"description": "AI and ML concepts"}
)

# Documents to index
documents = [
    "Transformers use self-attention mechanisms to process sequences in parallel",
    "BERT is a bidirectional transformer pretrained on masked language modeling",
    "GPT models are autoregressive transformers trained to predict the next token",
    "RAG combines retrieval from a knowledge base with language model generation",
    "Vector databases store embeddings and enable fast similarity search",
    "Fine-tuning adapts a pretrained model to a specific downstream task",
    "Embeddings are dense vector representations that capture semantic meaning",
    "Gradient descent minimizes loss by updating weights in the negative gradient direction",
    "Attention scores determine how much focus each token gives to other tokens",
    "LangChain is a framework for building applications with large language models",
]

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(documents).tolist()

# Add to ChromaDB
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"topic": "AI", "index": i} for i in range(len(documents))]
)

print(f"Added {collection.count()} documents to collection\n")

# ── EXPERIMENT 2: Querying ─────────────────────────────
print("=" * 60)
print("EXPERIMENT 2: Querying ChromaDB")
print("=" * 60)

queries = [
    "how does attention work in neural networks?",
    "what is the difference between BERT and GPT?",
    "how do I store and search vectors efficiently?",
]

for query in queries:
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances", "metadatas"]
    )
    
    print(f"\nQuery: '{query}'")
    print("Top 3 results:")
    for i, (doc, dist) in enumerate(zip(
        results["documents"][0], 
        results["distances"][0]
    )):
        # ChromaDB returns L2 distance — lower is better
        similarity = 1 / (1 + dist)
        print(f"  {i+1}. [{similarity:.3f}] {doc}")

# ── EXPERIMENT 3: Filtering with metadata ─────────────
print("\n" + "=" * 60)
print("EXPERIMENT 3: Metadata filtering")
print("=" * 60)

# Add documents with different metadata
collection2 = client.create_collection(name="mixed_docs")

mixed_docs = [
    ("Python is a high level programming language", {"category": "programming", "level": "basic"}),
    ("Neural networks are inspired by biological neurons", {"category": "ml", "level": "basic"}),
    ("Backpropagation computes gradients using the chain rule", {"category": "ml", "level": "advanced"}),
    ("Docker containers package applications with dependencies", {"category": "devops", "level": "basic"}),
    ("LoRA reduces trainable parameters using low rank decomposition", {"category": "ml", "level": "advanced"}),
]

collection2.add(
    documents=[d[0] for d in mixed_docs],
    embeddings=model.encode([d[0] for d in mixed_docs]).tolist(),
    ids=[f"mixed_{i}" for i in range(len(mixed_docs))],
    metadatas=[d[1] for d in mixed_docs]
)

# Query with metadata filter — only advanced ML topics
query = "how do neural networks learn?"
results = collection2.query(
    query_embeddings=[model.encode(query).tolist()],
    n_results=2,
    where={"$and": [{"category": {"$eq": "ml"}}, {"level": {"$eq": "advanced"}}]},
    include=["documents", "metadatas"]
)

print(f"\nQuery: '{query}'")
print("Filtered to: category=ml AND level=advanced")
print("Results:")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  • [{meta['category']} / {meta['level']}] {doc}")

# ── EXPERIMENT 4: Persistent storage ──────────────────
print("\n" + "=" * 60)
print("EXPERIMENT 4: Persistent ChromaDB")
print("=" * 60)

# This saves to disk — survives restarts
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get existing collection
try:
    persistent_collection = persistent_client.create_collection("persistent_test")
    persistent_collection.add(
        documents=["This document survives restarts"],
        embeddings=model.encode(["This document survives restarts"]).tolist(),
        ids=["persistent_1"]
    )
    print("Created persistent collection and added document")
except Exception:
    persistent_collection = persistent_client.get_collection("persistent_test")
    print("Loaded existing persistent collection")

print(f"Persistent collection has {persistent_collection.count()} documents")
print("Check your folder — you'll see a 'chroma_db' directory created")