"""
Unit tests for src/retrieval/.

Tests:
- BM25Retriever: index building, top-k retrieval, score ordering, determinism
- Corpus building: passage chunking, sample_id linkage
"""
import pytest

from src.retrieval.bm25_retriever import BM25Retriever, simple_tokenize
from src.retrieval.corpus import build_passage_corpus


class TestSimpleTokenize:
    """Tokenizer used by BM25."""

    def test_returns_list(self):
        result = simple_tokenize("hello world")
        assert isinstance(result, list)

    def test_lowercase(self):
        result = simple_tokenize("Hello World")
        assert all(t.islower() for t in result if t.isalpha())

    def test_empty_string(self):
        result = simple_tokenize("")
        assert isinstance(result, list)


class TestPassageCorpus:
    """Corpus construction from FinQA samples."""

    def test_corpus_nonempty(self, mini_corpus):
        assert len(mini_corpus) > 0, "Mini corpus should have at least 1 passage"

    def test_passage_has_required_fields(self, mini_corpus):
        passage = mini_corpus[0]
        assert "text" in passage
        assert "sample_id" in passage
        assert isinstance(passage["text"], str)
        assert len(passage["text"]) > 0

    def test_corpus_size_at_least_samples(self, mini_samples, mini_corpus):
        # Each sample produces at least 1 passage on average
        assert len(mini_corpus) >= len(mini_samples)


class TestBM25Retriever:
    """BM25 sparse retrieval."""

    @pytest.fixture(scope="class")
    def bm25(self, mini_corpus):
        """Build BM25 index once per test class."""
        return BM25Retriever(mini_corpus)

    def test_top_k_returns_at_most_k(self, bm25):
        results = bm25.retrieve("revenue increase", top_k=3)
        assert len(results) <= 3, "BM25 should return at most top_k results"

    def test_top_k_one(self, bm25):
        results = bm25.retrieve("revenue", top_k=1)
        assert len(results) <= 1

    def test_results_are_tuples(self, bm25):
        results = bm25.retrieve("financial", top_k=2)
        # Each result should be a (passage, score) tuple
        for item in results:
            assert len(item) == 2

    def test_score_descending_order(self, bm25):
        results = bm25.retrieve("revenue", top_k=5)
        scores = [score for _, score in results]
        # Scores should be in non-increasing order
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"BM25 results should be sorted by score descending"

    def test_returns_passage_dicts(self, bm25):
        results = bm25.retrieve("financial", top_k=2)
        for passage, score in results:
            assert isinstance(passage, dict)
            assert "text" in passage
            assert isinstance(score, (int, float))

    def test_query_with_no_match_still_returns_results(self, bm25):
        # Even nonsense queries return top-k (with low scores)
        results = bm25.retrieve("zzzzz xyz qqq", top_k=3)
        assert isinstance(results, list)


class TestRetrievalDeterminism:
    """BM25 should be deterministic — same query yields same results."""

    @pytest.fixture(scope="class")
    def bm25(self, mini_corpus):
        return BM25Retriever(mini_corpus)

    def test_query_repeatable(self, bm25):
        r1 = bm25.retrieve("revenue", top_k=3)
        r2 = bm25.retrieve("revenue", top_k=3)
        # Same passages in same order
        ids1 = [p.get("sample_id") for p, _ in r1]
        ids2 = [p.get("sample_id") for p, _ in r2]
        assert ids1 == ids2, "BM25 must be deterministic"
