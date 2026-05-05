"""
Dense retriever using sentence-transformers + FAISS.

Encodes passages and queries into 384-dim embeddings using all-MiniLM-L6-v2,
then uses FAISS IndexFlatIP for exact cosine similarity search (small corpus,
no need for approximate methods).
"""

from typing import List, Dict, Tuple
import numpy as np


class DenseRetriever:
    def __init__(self, passages: List[Dict],
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.passages = passages
        print(f"  Loading encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.dim = self.encoder.get_sentence_embedding_dimension()

        print(f"  Encoding {len(passages)} passages...")
        texts = [p['text'] for p in passages]
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,  # makes inner-product = cosine
        ).astype('float32')

        print(f"  Building FAISS index (dim={self.dim})...")
        self.index = faiss.IndexFlatIP(self.dim)  # IP = inner product
        self.index.add(embeddings)
        print(f"  Dense index built. {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Return top-k (passage_dict, similarity_score) tuples."""
        q_emb = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype('float32')
        scores, indices = self.index.search(q_emb, top_k)
        return [(self.passages[idx], float(score))
                for idx, score in zip(indices[0], scores[0])]

    def batch_retrieve(self, queries: List[str], top_k: int = 3, 
                       batch_size: int = 64) -> List[List[Tuple[Dict, float]]]:
        """Encode all queries at once for speed."""
        q_embs = self.encoder.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype('float32')
        scores, indices = self.index.search(q_embs, top_k)
        results = []
        for q_idx in range(len(queries)):
            results.append([
                (self.passages[indices[q_idx][k]], float(scores[q_idx][k]))
                for k in range(top_k)
            ])
        return results
