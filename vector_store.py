import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

CHROMA_PATH = "./chromadb_store"
COLLECTION_NAME = "pdf_knowledge_base"
EMBED_BATCH_SIZE = 32

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, docs: List[Dict[str, Any]]):
        texts = [d["text"] for d in docs]
        metadatas = [{"source": d["source"], "page": d["page"]} for d in docs]
        ids = [f"{d['source']}_p{d['page']}_{i}" for i, d in enumerate(docs)]
        embeddings = self._embed_texts(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f"Added {len(texts)} chunks to ChromaDB.")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i+EMBED_BATCH_SIZE]
            batch_embeds = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True).tolist()
            embeddings.extend(batch_embeds)
        return embeddings

    def similarity_search(self, query: str, top_k: int = 5):
        query_emb = self._embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist
            }
            for doc, meta, dist in zip(docs, metadatas, distances)
        ], query_emb

    def is_empty(self):
        return self.collection.count() == 0
