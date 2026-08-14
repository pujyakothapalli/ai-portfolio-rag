import os
import re
import json
import numpy as np
import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()

# ══════════════════════════════════════════════════════
# STEP 1: Document ingestion pipeline
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

def ingest_documents(documents, collection_name="naive_rag"):
    """Chunk, embed, and store documents in ChromaDB."""
    print(f"\n📥 Ingesting {len(documents)} documents...")
    
    collection = chroma.create_collection(collection_name)
    
    all_chunks = []
    all_embeddings = []
    all_ids = []
    all_metadata = []
    
    for doc_idx, (title, text) in enumerate(documents):
        chunks = semantic_chunking(text)
        print(f"   '{title}' → {len(chunks)} chunks")
        
        for chunk_idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            chunk_id = f"doc{doc_idx}_chunk{chunk_idx}"
            
            all_chunks.append(chunk)
            all_embeddings.append(embedding)
            all_ids.append(chunk_id)
            all_metadata.append({
                "source": title,
                "doc_idx": doc_idx,
                "chunk_idx": chunk_idx
            })
    
    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadata
    )
    
    print(f"   ✅ Total chunks stored: {collection.count()}")
    return collection

# ══════════════════════════════════════════════════════
# STEP 2: Retrieval
# ══════════════════════════════════════════════════════

def retrieve(query, collection, top_k=3):
    """Retrieve top-k relevant chunks for a query."""
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    
    retrieved = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0]
    ):
        similarity = 1 / (1 + dist)
        retrieved.append({
            "content": doc,
            "similarity": similarity,
            "source": meta["source"]
        })
    
    return retrieved

# ══════════════════════════════════════════════════════
# STEP 3: Generation
# ══════════════════════════════════════════════════════

def generate(query, retrieved_chunks):
    """Generate answer using retrieved context."""
    
    # Format context
    context = "\n\n".join([
        f"Source: {chunk['source']}\n{chunk['content']}"
        for chunk in retrieved_chunks
    ])
    
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        temperature=0,
        system="""You are a helpful assistant that answers questions 
based ONLY on the provided context. 
If the answer is not in the context, say "I don't have enough information to answer this."
Always cite your source.""",
        messages=[{"role": "user", "content": f"""Context:
{context}

Question: {query}

Answer based only on the context above:"""}]
    )
    
    return response.content[0].text

# ══════════════════════════════════════════════════════
# STEP 4: Full RAG pipeline
# ══════════════════════════════════════════════════════

def rag_query(query, collection, top_k=3, verbose=True):
    """Complete RAG pipeline: retrieve → generate."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
    
    # Retrieve
    retrieved = retrieve(query, collection, top_k)
    
    if verbose:
        print(f"\n📚 Retrieved {len(retrieved)} chunks:")
        for i, chunk in enumerate(retrieved):
            print(f"   {i+1}. [{chunk['similarity']:.3f}] "
                  f"({chunk['source']}) {chunk['content'][:80]}...")
    
    # Generate
    answer = generate(query, retrieved)
    
    if verbose:
        print(f"\n🤖 Answer:\n{answer}")
    
    return answer, retrieved

# ══════════════════════════════════════════════════════
# TEST DOCUMENTS
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
# RUN THE PIPELINE
# ══════════════════════════════════════════════════════

print("🚀 Building naive RAG pipeline from scratch")
print("No LangChain. Pure Python + ChromaDB + Anthropic.\n")

# Ingest
collection = ingest_documents(documents)

# Test queries
test_queries = [
    "How does RAG reduce hallucinations?",
    "What are the limitations of large language models?",
    "How does chunk size affect retrieval quality?",
    "What is the capital of France?",  # Out of domain — tests "I don't know"
]

for query in test_queries:
    answer, retrieved = rag_query(query, collection)

# ══════════════════════════════════════════════════════
# FAILURE ANALYSIS
# ══════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("FAILURE ANALYSIS — Where does naive RAG break?")
print("=" * 60)

failure_queries = [
    "Compare vector databases and explain which is best",
    "What should I do first when building a RAG system?",
    "Summarize everything about embeddings and chunking together",
]

print("\nThese queries will show naive RAG's weaknesses:\n")
for query in failure_queries:
    print(f"Query: '{query}'")
    retrieved = retrieve(query, collection, top_k=3)
    print(f"Top retrieved source: {retrieved[0]['source']} "
          f"(similarity: {retrieved[0]['similarity']:.3f})")
    print(f"Issue: ", end="")
    if "Compare" in query:
        print("Comparative query — retriever returns one source, misses the comparison")
    elif "first" in query:
        print("Procedural query — no step-by-step content in our docs")
    elif "together" in query:
        print("Multi-topic query — retriever can only focus on one topic at a time")
    print()