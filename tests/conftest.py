"""
Pytest fixtures shared across test files.

Fixtures defined here are automatically available to any test in tests/.
"""
import json
import os
import sys

import pytest

# Make src importable from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def mini_samples():
    """A tiny stratified sample for fast end-to-end testing."""
    sample_path = "data/samples/finqa_10_demo.json"
    with open(sample_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def mini_corpus(mini_samples):
    """Build a small passage corpus from mini_samples for retrieval tests."""
    from src.retrieval.corpus import build_passage_corpus
    return build_passage_corpus(mini_samples)


@pytest.fixture
def gold_predictions():
    """A canonical set of (gold, prediction) pairs covering all answer types."""
    return [
        # (gold, prediction, expected_format_match, expected_numeric_close)
        ("yes", "yes", True, False),  # boolean exact
        ("no", "yes", True, False),  # boolean format match, wrong answer
        ("12.5%", "12.4%", True, True),  # percentage close
        ("12.5%", "1000%", True, False),  # percentage format match, far
        ("100", "99", True, True),  # numeric close
        ("100", "yes", False, False),  # numeric vs boolean (format mismatch)
    ]
