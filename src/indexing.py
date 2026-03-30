"""
Vector Store Manager for the Domain Q&A RAG Assistant.

Handles embedding generation (via sentence-transformers) and
FAISS index operations (create, save, load, search).

All model names.and retrieval parameters are read from config.yaml
so nothing is hardcoded.

Professional Commit Message:
    refactor(indexing): read model name and params from config.yaml
"""

import faiss
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

from src.config import cfg


class Indexer:
    def __init__(
        self,
        model_name: str = None,
        index_path: str = None,
    ):
        # Fall back to config values if not explicitly provided
        model_name = model_name or cfg.embedding.model
        index_path = index_path or cfg.retrieval.index_path

        # Use GPU if available for faster embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.index_path = index_path
        self.index = None
        self.chunks: List[str] = []

    def create_vector_store(self, chunks: List[str]):
        """Embeds chunks and creates or updates a FAISS index."""
        if not chunks:
            return

        # Batch processing with optimal batch size from config
        embeddings = self.model.encode(
            chunks,
            batch_size=cfg.embedding.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        dimension = embeddings.shape[1]

        if self.index is None:
            # Try to load existing index first if we haven't already
            if os.path.exists(self.index_path):
                self.load_index()

            # If still None (no file existed), create new
            if self.index is None:
                self.index = faiss.IndexFlatL2(dimension)

        # Check if dimension matches
        if self.index.d != dimension:
            # If dimensions mismatch (different model used previously?), reset
            print("Dimension mismatch, resetting index.")
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []

        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(chunks)

    def save_index(self):
        """Saves the index and chunks to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.index_path + ".chunks", "w", encoding="utf-8") as f:
                for chunk in self.chunks:
                    f.write(chunk.replace("\n", " ") + "\n")

    def load_index(self):
        """Loads the index and chunks from disk."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.index_path + ".chunks"):
                with open(self.index_path + ".chunks", "r", encoding="utf-8") as f:
                    self.chunks = [line.strip() for line in f.readlines()]
        else:
            print("Index not found.")

    def search(self, query: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Searches the FAISS index for the k nearest chunks.

        Returns:
            List of (chunk_text, l2_distance) tuples sorted by relevance.
        """
        if k is None:
            k = cfg.retrieval.top_k

        if not self.index:
            return []

        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding).astype("float32"), k)

        results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(D[0][i])))
        return results
