import os
import re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# ══════════════════════════════════════════════════════
# DOCUMENTS — same as Day 3
# ══════════════════════════════════════════════════════

documents = [
    ("RAG Overview", """
Retrieval-Augmented Generation (RAG) is an AI framework that enhances language 
model outputs by retrieving relevant information from external knowledge bases. 
RAG reduces hallucinations by grounding responses in retrieved facts. The system 
consists of two components: a retriever that finds relevant documents and a 
generator that produces answers conditioned on those documents.
    """),
    ("Vector Databases", """
Vector databases store high-dimensional embeddings and enable fast similarity 
search using approximate nearest neighbor algorithms. Popular vector databases 
include Pinecone, Weaviate, Qdrant, and ChromaDB. They support metadata filtering 
which allows combining semantic search with structured queries. Vector databases 
are essential infrastructure for RAG systems and semantic search applications.
    """),
    ("LLM Limitations", """
Large language models suffer from several key limitations. Hallucination occurs 
when models generate plausible but factually incorrect information. Knowledge 
cutoffs mean models lack information about recent events. Context window limits 
restrict how much text can be processed at once. Models can also exhibit biases 
present in their training data and struggle with precise numerical reasoning.
    """),
    ("Embeddings", """
Text embeddings are dense vector representations that capture semantic meaning. 
Similar texts have embeddings that are close together in vector space, measured 
by cosine similarity. Embedding models like sentence-transformers are trained on 
large corpora to produce meaningful representations. The quality of embeddings 
directly impacts retrieval quality in RAG systems.
    """),
    ("Chunking Strategies", """
Document chunking determines how text is split before embedding. Fixed-size 
chunking splits by word or character count but can break semantic units. 
Recursive chunking respects document structure like paragraphs and sentences. 
Semantic chunking uses embedding similarity to detect topic boundaries. 
Chunk size affects retrieval precision — smaller chunks are more precise 
but may lack context, while larger chunks provide more context but reduce precision.
    """),
]

# ══════════════════════════════════════════════════════
# SETUP — chunk and prepare all retrievers
# ══════════════════════════════════════════════════════

def semantic_chunking(text, threshold=0.3):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return [text]
    embeddings = model.encode(sentences)
    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Build chunk store
print("Building chunk store...")
all_chunks = []
all_sources = []

for title, text in documents:
    chunks = semantic_chunking(text)
    for chunk in chunks:
        all_chunks.append(chunk)
        all_sources.append(title)

print(f"Total chunks: {len(all_chunks)}")

# Build ChromaDB for dense retrieval
chroma = chromadb.Client()
collection = chroma.create_collection("hybrid_rag")
embeddings = model.encode(all_chunks).tolist()
collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(all_chunks))],
    metadatas=[{"source": s} for s in all_sources]
)

# Build BM25 for sparse retrieval
tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)
print("Retrievers ready.\n")

# ══════════════════════════════════════════════════════
# THREE RETRIEVAL STRATEGIES
# ══════════════════════════════════════════════════════

def dense_retrieval(query, top_k=5):
    """Pure semantic/vector search."""
    query_emb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    retrieved = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        retrieved.append({
            "content": doc,
            "score": 1 / (1 + dist),
            "source": meta["source"],
            "method": "dense"
        })
    return retrieved

def sparse_retrieval(query, top_k=5):
    """Pure BM25 keyword search."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    retrieved = []
    for idx in top_indices:
        if scores[idx] > 0:
            retrieved.append({
                "content": all_chunks[idx],
                "score": float(scores[idx]),
                "source": all_sources[idx],
                "method": "sparse"
            })
    return retrieved

def hybrid_retrieval(query, top_k=5, dense_weight=0.7, sparse_weight=0.3):
    """Combine dense + sparse with weighted scoring."""
    # Get candidates from both
    dense_results = dense_retrieval(query, top_k=top_k*2)
    sparse_results = sparse_retrieval(query, top_k=top_k*2)
    
    # Normalize scores to 0-1
    def normalize(results):
        if not results:
            return results
        scores = [r["score"] for r in results]
        max_s, min_s = max(scores), min(scores)
        if max_s == min_s:
            return results
        for r in results:
            r["score"] = (r["score"] - min_s) / (max_s - min_s)
        return results
    
    dense_results = normalize(dense_results)
    sparse_results = normalize(sparse_results)
    
    # Combine scores
    combined = {}
    for r in dense_results:
        key = r["content"][:50]
        combined[key] = {
            **r,
            "final_score": dense_weight * r["score"],
            "dense_score": r["score"],
            "sparse_score": 0
        }
    
    for r in sparse_results:
        key = r["content"][:50]
        if key in combined:
            combined[key]["final_score"] += sparse_weight * r["score"]
            combined[key]["sparse_score"] = r["score"]
        else:
            combined[key] = {
                **r,
                "final_score": sparse_weight * r["score"],
                "dense_score": 0,
                "sparse_score": r["score"]
            }
    
    # Sort by final score
    results = sorted(combined.values(), 
                    key=lambda x: x["final_score"], 
                    reverse=True)
    return results[:top_k]

# ══════════════════════════════════════════════════════
# BENCHMARK: Dense vs Sparse vs Hybrid
# ══════════════════════════════════════════════════════

print("=" * 65)
print("BENCHMARK: Dense vs Sparse vs Hybrid Retrieval")
print("=" * 65)

# These queries test different retrieval strengths
test_queries = [
    {
        "query": "How does RAG reduce hallucinations?",
        "type": "SEMANTIC",
        "note": "Dense should win — meaning matters more than keywords"
    },
    {
        "query": "Pinecone Weaviate Qdrant ChromaDB",
        "type": "KEYWORD",
        "note": "Sparse should win — exact keyword match"
    },
    {
        "query": "approximate nearest neighbor similarity search embeddings",
        "type": "MIXED",
        "note": "Hybrid should win — both keywords and semantics matter"
    },
    {
        "query": "what breaks when context is too long?",
        "type": "SEMANTIC",
        "note": "Tests semantic understanding of 'context window limits'"
    },
]

for test in test_queries:
    query = test["query"]
    print(f"\n── Query: '{query}'")
    print(f"   Type: {test['type']} | {test['note']}")
    
    dense = dense_retrieval(query, top_k=1)
    sparse = sparse_retrieval(query, top_k=1)
    hybrid = hybrid_retrieval(query, top_k=1)
    
    print(f"\n   Dense  [{dense[0]['score']:.3f}]: "
          f"({dense[0]['source']}) {dense[0]['content'][:80]}...")
    
    if sparse:
        print(f"   Sparse [{sparse[0]['score']:.3f}]: "
              f"({sparse[0]['source']}) {sparse[0]['content'][:80]}...")
    else:
        print(f"   Sparse [0.000]: No keyword matches found")
    
    print(f"   Hybrid [{hybrid[0]['final_score']:.3f}]: "
          f"({hybrid[0]['source']}) {hybrid[0]['content'][:80]}...")

# ══════════════════════════════════════════════════════
# FAILURE CASE FROM DAY 3 — fixed by hybrid?
# ══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("DAY 3 FAILURE CASE: 'What are the limitations of LLMs?'")
print("Does hybrid search retrieve MORE relevant chunks?")
print("=" * 65)

query = "What are the limitations of large language models?"

dense_results = dense_retrieval(query, top_k=3)
hybrid_results = hybrid_retrieval(query, top_k=3)

print(f"\nDense retrieval top 3:")
for r in dense_results:
    print(f"  [{r['score']:.3f}] ({r['source']}) {r['content'][:80]}...")

print(f"\nHybrid retrieval top 3:")
for r in hybrid_results:
    print(f"  [{r['final_score']:.3f}] ({r['source']}) {r['content'][:80]}...")