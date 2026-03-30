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
import json
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Union, Dict

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
        self.bm25_path = index_path.replace(".bin", "_bm25.pkl") if index_path else cfg.retrieval.bm25_path
        self.index = None
        self.bm25 = None
        self.chunks: List[Dict[str, str]] = []

    def create_vector_store(self, chunks: List[Union[str, dict]]):
        """Embeds chunks, creates FAISS dense index and BM25 sparse index."""
        if not chunks:
            return

        # Normalize chunks to new Dict format
        parsed_chunks = []
        for c in chunks:
            if isinstance(c, str):
                parsed_chunks.append({"text": c, "parent_text": c})
            else:
                parsed_chunks.append(c)

        texts_to_embed = [c["text"] for c in parsed_chunks]

        # Batch processing with optimal batch size from config
        embeddings = self.model.encode(
            texts_to_embed,
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
            print("Dimension mismatch, resetting index.")
            self.index = faiss.IndexFlatL2(dimension)
            self.chunks = []
            self.bm25 = None

        # Add to FAISS
        self.index.add(np.array(embeddings).astype("float32"))
        self.chunks.extend(parsed_chunks)
        
        # Build/Update BM25
        all_texts = [c["text"] for c in self.chunks]
        tokenized_corpus = [doc.lower().split() for doc in all_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_index(self):
        """Saves the FAISS index, BM25 index, and chunks to disk."""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            with open(self.index_path + ".chunks", "w", encoding="utf-8") as f:
                json.dump(self.chunks, f)
            if self.bm25:
                with open(self.bm25_path, "wb") as f:
                    pickle.dump(self.bm25, f)

    def load_index(self):
        """Loads indices and chunks from disk."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if os.path.exists(self.index_path + ".chunks"):
                with open(self.index_path + ".chunks", "r", encoding="utf-8") as f:
                    try:
                        self.chunks = json.load(f)
                    except json.JSONDecodeError:
                        # Backwards compatibility for old line-by-line format
                        f.seek(0)
                        self.chunks = [{"text": line.strip(), "parent_text": line.strip()} for line in f.readlines()]
            if os.path.exists(self.bm25_path):
                with open(self.bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
            elif self.chunks:
                # Rebuild BM25 if missing
                tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
                self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print("Index not found.")

    def search(self, query: str, k: int = None) -> List[Tuple[Dict[str, str], float]]:
        """
        Searches the FAISS index for the k nearest chunks.

        Returns:
            List of (chunk_dict, l2_distance) tuples sorted by relevance.
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
