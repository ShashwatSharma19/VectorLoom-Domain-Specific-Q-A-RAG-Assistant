"""
Hybrid Retriever for VectorLoom using Reciprocal Rank Fusion and Cross-Encoders.

Professional Commit Message:
    feat(retrieval): implement Hybrid Search (FAISS + BM25) with Cross-Encoder reranking
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder

from src.config import cfg
from src.indexing import Indexer


class HybridRetriever:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        
        # Initialize Cross-Encoder (lazy download if not present)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(cfg.retrieval.reranker_model, device=device)

    def search(self, query: str, top_k: int = None, final_k: int = None) -> List[Tuple[Dict[str, str], float]]:
        """
        Executes hybrid search and cross-encoder reranking.
        """
        top_k = top_k or cfg.retrieval.top_k
        final_k = final_k or cfg.retrieval.final_k
        
        if not self.indexer.index or not self.indexer.bm25:
            return []

        # ── 1. FAISS Search (Dense) ──
        dense_results = self.indexer.search(query, k=top_k)
        
        # ── 2. BM25 Search (Sparse) ──
        tokenized_query = query.lower().split()
        bm25_scores = self.indexer.bm25.get_scores(tokenized_query)
        # Get top indices from numpy array
        top_n_sparse_idx = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_results = [(self.indexer.chunks[idx], bm25_scores[idx]) for idx in top_n_sparse_idx]

        # ── 3. Reciprocal Rank Fusion (RRF) ──
        # RRF formula: 1 / (k_rrf + rank) where k_rrf is typically 60
        k_rrf = 60
        ranks = {}
        
        for rank, (chunk, _) in enumerate(dense_results):
            cid = id(chunk)
            ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
            
        for rank, (chunk, _) in enumerate(sparse_results):
            cid = id(chunk)
            ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (k_rrf + rank + 1)
            
        # Sort by best combined RRF score
        sorted_cids = sorted(ranks.keys(), key=lambda x: ranks[x], reverse=True)
        
        # Re-build candidate objects pool by tracking the Python object ids
        pool = {id(c): c for c, _ in dense_results + sparse_results}
        top_candidates = [pool[cid] for cid in sorted_cids[:top_k]]

        # ── 4. Cross-Encoder Reranking ──
        if not top_candidates:
            return []
            
        # We rerank against the precise `text` child chunk, not the inflated `parent` context,
        # because the MS Macro Cross-Encoder is trained on shorter precise passages!
        cross_inputs = [[query, chunk["text"]] for chunk in top_candidates]
        cross_scores = self.reranker.predict(cross_inputs)
        
        # Zip candidates with their new absolute relevance scores
        reranked = sorted(zip(top_candidates, cross_scores), key=lambda x: float(x[1]), reverse=True)
        
        return reranked[:final_k]
