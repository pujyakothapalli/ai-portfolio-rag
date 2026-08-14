import re
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Sample document ────────────────────────────────────
document = """
Artificial intelligence has transformed the way we interact with technology. 
From virtual assistants to recommendation systems, AI is embedded in our daily lives.
The field has seen rapid growth over the past decade, driven by advances in computing 
power and the availability of large datasets.

Machine learning is a subset of artificial intelligence that focuses on building 
systems that learn from data. Instead of being explicitly programmed, these systems 
identify patterns and make decisions with minimal human intervention. Supervised 
learning, unsupervised learning, and reinforcement learning are the three main 
paradigms in machine learning.

Deep learning represents a further specialization within machine learning, using 
neural networks with many layers to model complex patterns. Convolutional neural 
networks excel at image recognition tasks, while recurrent neural networks handle 
sequential data like text and time series. The transformer architecture has 
revolutionized natural language processing since its introduction in 2017.

Natural language processing enables computers to understand and generate human 
language. Applications include machine translation, sentiment analysis, question 
answering, and text summarization. Large language models like GPT and Claude have 
demonstrated remarkable capabilities in generating coherent and contextually 
appropriate text across diverse domains.

Retrieval augmented generation combines the strengths of retrieval systems and 
generative models. By grounding language model outputs in retrieved documents, 
RAG systems reduce hallucinations and improve factual accuracy. The architecture 
consists of a retriever that finds relevant documents and a generator that produces 
responses conditioned on those documents.
"""

# ══════════════════════════════════════════════════════
# STRATEGY 1: Fixed size chunking
# ══════════════════════════════════════════════════════

def fixed_size_chunking(text, chunk_size=100, overlap=20):
    """Split by word count with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ══════════════════════════════════════════════════════
# STRATEGY 2: Recursive character chunking
# ══════════════════════════════════════════════════════

def recursive_chunking(text, chunk_size=500, overlap=50):
    """Split by paragraphs first, then sentences, then words."""
    # Try splitting by paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph keeps us under limit
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += " " + para
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            if len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                current_chunk = overlap_text + " " + para
            else:
                current_chunk = para
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# ══════════════════════════════════════════════════════
# STRATEGY 3: Semantic chunking
# ══════════════════════════════════════════════════════

def semantic_chunking(text, threshold=0.3):
    """Split where meaning changes significantly."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return [text]
    
    # Embed all sentences
    embeddings = model.encode(sentences)
    
    # Find where similarity drops (meaning changes)
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Similarity between consecutive sentences
        sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        
        if sim < threshold:
            # Big topic shift — start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ══════════════════════════════════════════════════════
# COMPARE ALL THREE
# ══════════════════════════════════════════════════════

print("=" * 60)
print("CHUNKING STRATEGY COMPARISON")
print("=" * 60)

strategies = [
    ("Fixed Size (100 words, 20 overlap)", fixed_size_chunking(document)),
    ("Recursive Character", recursive_chunking(document)),
    ("Semantic", semantic_chunking(document)),
]

for name, chunks in strategies:
    print(f"\n── {name} ──")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Avg chunk size: {sum(len(c.split()) for c in chunks) // len(chunks)} words")
    print(f"   Smallest chunk: {min(len(c.split()) for c in chunks)} words")
    print(f"   Largest chunk:  {max(len(c.split()) for c in chunks)} words")
    print(f"\n   Preview of chunk 1:")
    print(f"   {chunks[0][:150]}...")

# ══════════════════════════════════════════════════════
# RETRIEVAL QUALITY TEST
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("RETRIEVAL QUALITY: Which chunking finds the right answer?")
print("=" * 60)

def retrieve(query, chunks, top_k=2):
    """Simple retrieval using cosine similarity."""
    query_emb = model.encode(query)
    chunk_embs = model.encode(chunks)
    
    scores = []
    for i, chunk_emb in enumerate(chunk_embs):
        score = np.dot(query_emb, chunk_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)
        )
        scores.append((score, i, chunks[i]))
    
    scores.sort(reverse=True)
    return scores[:top_k]

query = "How does RAG reduce hallucinations?"

print(f"\nQuery: '{query}'\n")

for name, chunks in strategies:
    results = retrieve(query, chunks)
    print(f"── {name} ──")
    for score, idx, chunk in results:
        print(f"   Score: {score:.3f} | Chunk {idx+1}: {chunk[:120]}...")
    print()