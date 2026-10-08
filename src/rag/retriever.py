import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

class HybridRetriever:
    def __init__(self, collection_name: str = "rag_store"):
        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = collection_name
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.bm25 = None
        self.collection = None
        self._load_or_create()

    def _load_or_create(self):
        try:
            self.collection = self.chroma.get_collection(self.collection_name)
            print(f"Loaded existing collection: {self.collection.count()} chunks")
        except:
            self.collection = self.chroma.create_collection(self.collection_name)
            print("Created new collection")

    def add_documents(self, chunks: List[str], source: str):
        if not chunks:
            return
        embeddings = model.encode(chunks).tolist()
        start_id = self.collection.count()
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"chunk_{start_id + i}" for i in range(len(chunks))],
            metadatas=[{"source": source} for _ in chunks]
        )
        # Rebuild BM25 index
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.chunks = all_data["documents"]
        self.sources = [m["source"] for m in all_data["metadatas"]]
        self.bm25 = BM25Okapi([c.lower().split() for c in self.chunks])
        print(f"Added {len(chunks)} chunks. Total: {self.collection.count()}")

    def dense_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "distances", "metadatas"]
        )
        return [
            {"content": doc, "score": 1/(1+dist),
             "source": meta["source"], "method": "dense"}
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0]
            )
        ]

    def hybrid_retrieve(self, query: str, top_k: int = 5,
                        dense_weight: float = 0.7) -> List[Dict]:
        if not self.chunks:
            return self.dense_retrieve(query, top_k)

        dense = self.dense_retrieve(query, top_k * 2)
        sparse_scores = self.bm25.get_scores(query.lower().split())
        top_sparse_idx = np.argsort(sparse_scores)[::-1][:top_k * 2]

        sparse = [
            {"content": self.chunks[i], "score": float(sparse_scores[i]),
             "source": self.sources[i], "method": "sparse"}
            for i in top_sparse_idx if sparse_scores[i] > 0
        ]

        def normalize(results):
            if not results:
                return results
            scores = [r["score"] for r in results]
            mx, mn = max(scores), min(scores)
            if mx == mn:
                return results
            for r in results:
                r["score"] = (r["score"] - mn) / (mx - mn)
            return results

        dense = normalize(dense)
        sparse = normalize(sparse)

        combined = {}
        for r in dense:
            key = r["content"][:60]
            combined[key] = {**r, "final_score": dense_weight * r["score"]}

        sparse_weight = 1 - dense_weight
        for r in sparse:
            key = r["content"][:60]
            if key in combined:
                combined[key]["final_score"] += sparse_weight * r["score"]
            else:
                combined[key] = {**r, "final_score": sparse_weight * r["score"]}

        return sorted(combined.values(),
                      key=lambda x: x["final_score"], reverse=True)[:top_k]

    def reset(self):
        try:
            self.chroma.delete_collection(self.collection_name)
        except:
            pass
        self.collection = self.chroma.create_collection(self.collection_name)
        self.chunks, self.sources, self.bm25 = [], [], None