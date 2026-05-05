"""
BM25 sparse retriever for FinQA RAG.

BM25 ranks passages by exact word overlap with the query, with TF-IDF style
weighting. Fast, no GPU needed, strong baseline for keyword-heavy domains.
"""

import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric. Keeps numbers as tokens."""
    return re.findall(r'\w+', text.lower())


class BM25Retriever:
    def __init__(self, passages: List[Dict]):
        """
        Args:
            passages: list of {'passage_id': ..., 'text': ..., ...} dicts
        """
        self.passages = passages
        print(f"  Tokenizing {len(passages)} passages...")
        tokenized_corpus = [simple_tokenize(p['text']) for p in passages]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"  BM25 index built.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Return top-k (passage_dict, score) tuples ranked by BM25 score."""
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Get indices of top-k highest scores
        top_indices = sorted(range(len(scores)),
                             key=lambda i: scores[i],
                             reverse=True)[:top_k]
        return [(self.passages[i], float(scores[i])) for i in top_indices]

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Tuple[Dict, float]]]:
        """Retrieve top-k for each query in a list."""
        return [self.retrieve(q, top_k) for q in queries]
