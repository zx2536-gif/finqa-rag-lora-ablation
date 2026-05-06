"""
Integration tests: validate that pipeline components work together.

These tests use a 10-sample mini dataset for speed. They verify:
- Prompt building does not crash on real samples
- Context construction works on real samples
- Metrics evaluation runs on (predictions, golds) pairs

These are smoke tests — they confirm the pipeline runs, not that it
achieves any specific accuracy.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import evaluate_predictions
from src.utils.data_utils import build_context, build_prompt, get_gold_answer


class TestPromptBuilding:
    """Verify prompt construction for FinQA samples."""

    def test_build_prompt_returns_string(self, mini_samples):
        prompt = build_prompt(mini_samples[0])
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_prompt_contains_question(self, mini_samples):
        sample = mini_samples[0]
        prompt = build_prompt(sample)
        # First 3 words of question should appear in prompt
        question = sample["qa"]["question"]
        first_words = " ".join(question.split()[:3]).lower()
        assert first_words in prompt.lower(), \
            f"Prompt should contain question text"

    def test_build_context_returns_string(self, mini_samples):
        ctx = build_context(mini_samples[0])
        assert isinstance(ctx, str)

    def test_build_prompt_works_on_all_mini_samples(self, mini_samples):
        # No sample should cause a crash
        for sample in mini_samples:
            prompt = build_prompt(sample)
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestGoldExtraction:
    """get_gold_answer extracts the answer field."""

    def test_returns_string(self, mini_samples):
        for sample in mini_samples:
            gold = get_gold_answer(sample)
            assert isinstance(gold, str)


class TestEvaluation:
    """Verify metric evaluation pipeline runs end-to-end."""

    def test_perfect_predictions_yield_high_f1(self, mini_samples):
        """Perfect predictions (predict gold) should give high F1."""
        predictions = [get_gold_answer(s) for s in mini_samples]
        golds = [get_gold_answer(s) for s in mini_samples]
        results = evaluate_predictions(predictions, golds)
        
        assert isinstance(results, dict)
        f1 = results.get("overall", {}).get("f1", 0)
        # Perfect predictions should give F1 close to 1.0
        assert f1 > 0.9, f"Perfect predictions should give F1 > 0.9, got {f1}"

    def test_all_wrong_predictions_yield_low_f1(self, mini_samples):
        """Constant nonsense predictions should give near-zero F1."""
        predictions = ["xyzzy nonsense" for _ in mini_samples]
        golds = [get_gold_answer(s) for s in mini_samples]
        results = evaluate_predictions(predictions, golds)
        
        f1 = results.get("overall", {}).get("f1", 0)
        # Wrong predictions should give very low F1
        assert f1 < 0.3, f"Wrong predictions should give low F1, got {f1}"

    def test_evaluate_returns_overall_section(self, mini_samples):
        predictions = [get_gold_answer(s) for s in mini_samples]
        golds = [get_gold_answer(s) for s in mini_samples]
        results = evaluate_predictions(predictions, golds)
        
        assert "overall" in results
        # Common keys we expect
        overall = results["overall"]
        assert isinstance(overall, dict)


class TestEndToEndSmokeTest:
    """Smoke test: required artifacts exist for inference."""

    def test_finqa_500_dataset_exists(self):
        """The 500-sample stratified set should exist after preprocessing."""
        path = "data/processed/finqa_500.json"
        if not os.path.exists(path):
            pytest.skip(f"{path} not present (run data/preprocess.py first)")
        
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 500, f"Expected 500 samples, got {len(data)}"

    def test_mini_demo_exists(self):
        """The 10-sample demo set should exist for quick testing."""
        path = "data/samples/finqa_10_demo.json"
        if not os.path.exists(path):
            pytest.skip(f"{path} not present")
        
        with open(path) as f:
            data = json.load(f)
        assert len(data) > 0
