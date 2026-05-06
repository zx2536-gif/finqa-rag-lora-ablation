"""
Unit tests for src/evaluation/metrics.py.

Tests core evaluation functions matching the actual API:
- normalize_answer: text normalization (lowercase, remove articles/punctuation)
- f1_score: token-level F1 between prediction and gold
- lcs_length: longest common subsequence (helper for ROUGE-L)
- rouge_l: ROUGE-L score
- classify_answer_format: categorize as boolean/percentage/numeric/text
- format_match: prediction-gold category match (returns 0.0 or 1.0)
- extract_number: parse numeric value from string
- numeric_tolerance_match: relative-error correctness (returns 0.0 or 1.0)
"""
import pytest

from src.evaluation.metrics import (
    classify_answer_format,
    extract_number,
    f1_score,
    format_match,
    lcs_length,
    normalize_answer,
    numeric_tolerance_match,
    rouge_l,
)


class TestNormalizeAnswer:
    """normalize_answer: lowercase, remove articles, strip punctuation."""

    def test_lowercase(self):
        assert normalize_answer("YES") == "yes"

    def test_strip_whitespace(self):
        assert normalize_answer("  yes  ") == "yes"

    def test_remove_punctuation(self):
        assert normalize_answer("yes!") == "yes"
        assert normalize_answer("12.5%") == "125"  # punctuation removed

    def test_remove_articles(self):
        # "the cat" → "cat" (article removed)
        result = normalize_answer("the cat")
        assert result == "cat"

    def test_empty_or_none(self):
        assert normalize_answer("") == ""
        assert normalize_answer(None) == ""


class TestF1Score:
    """Token-level F1 score."""

    def test_perfect_match(self):
        assert f1_score("yes", "yes") == pytest.approx(1.0)

    def test_complete_mismatch(self):
        assert f1_score("yes", "no") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # "the quick fox" vs "the slow fox" — share "the" and "fox"
        # After normalization: "quick fox" vs "slow fox" (article removed)
        score = f1_score("the quick fox", "the slow fox")
        assert 0.0 < score < 1.0

    def test_case_insensitive(self):
        assert f1_score("Yes", "yes") == pytest.approx(1.0)

    def test_both_empty_returns_one(self):
        # By implementation, both empty after normalization → 1.0
        assert f1_score("", "") == pytest.approx(1.0)

    def test_one_empty_returns_zero(self):
        assert f1_score("yes", "") == pytest.approx(0.0)
        assert f1_score("", "yes") == pytest.approx(0.0)


class TestLcsLength:
    """Longest Common Subsequence helper."""

    def test_identical_lists(self):
        assert lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_no_common(self):
        assert lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_subsequence(self):
        # "a c" is subsequence of "a b c"
        assert lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_empty_inputs(self):
        assert lcs_length([], ["a"]) == 0
        assert lcs_length(["a"], []) == 0
        assert lcs_length([], []) == 0


class TestRougeL:
    """ROUGE-L score (LCS-based)."""

    def test_perfect_match(self):
        assert rouge_l("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)

    def test_complete_mismatch(self):
        assert rouge_l("yes", "no") == pytest.approx(0.0)

    def test_subsequence_credit(self):
        # Some overlap should yield non-zero score
        score = rouge_l("the quick brown fox", "quick fox")
        assert score > 0.0


class TestAnswerFormatClassification:
    """classify_answer_format identifies answer category."""

    def test_boolean_yes(self):
        assert classify_answer_format("yes") == "boolean"

    def test_boolean_no(self):
        assert classify_answer_format("no") == "boolean"

    def test_boolean_caps(self):
        assert classify_answer_format("Yes") == "boolean"
        assert classify_answer_format("NO") == "boolean"

    def test_percentage_simple(self):
        assert classify_answer_format("12.5%") == "percentage"

    def test_percentage_negative(self):
        assert classify_answer_format("-4%") == "percentage"

    def test_numeric_integer(self):
        assert classify_answer_format("123") == "numeric"

    def test_numeric_decimal(self):
        assert classify_answer_format("12.5") == "numeric"

    def test_numeric_negative(self):
        assert classify_answer_format("-558") == "numeric"

    def test_text_arbitrary(self):
        # Long arbitrary text should not be classified as a numeric type
        result = classify_answer_format("entergy wholesale commodities")
        assert result not in ("boolean", "percentage", "numeric")


class TestFormatMatch:
    """format_match: prediction-gold category match (returns 0.0 or 1.0)."""

    def test_same_category_match(self):
        # Both percentages → format match = 1.0
        assert format_match("12.5%", "10%") == pytest.approx(1.0)

    def test_different_category_mismatch(self):
        # Percentage vs boolean → mismatch
        assert format_match("12.5%", "yes") == pytest.approx(0.0)

    def test_boolean_match(self):
        assert format_match("yes", "no") == pytest.approx(1.0)

    def test_numeric_match(self):
        assert format_match("100", "200") == pytest.approx(1.0)


class TestExtractNumber:
    """extract_number parses numeric values."""

    def test_integer(self):
        assert extract_number("123") == pytest.approx(123.0)

    def test_decimal(self):
        assert extract_number("12.5") == pytest.approx(12.5)

    def test_negative(self):
        assert extract_number("-558") == pytest.approx(-558.0)

    def test_percentage_format(self):
        # "12.5%" → 12.5
        result = extract_number("12.5%")
        assert result is not None
        assert result == pytest.approx(12.5)

    def test_no_number_returns_none(self):
        assert extract_number("yes") is None
        assert extract_number("hello world") is None


class TestNumericTolerance:
    """numeric_tolerance_match: returns 0.0 or 1.0 based on relative error."""

    def test_exact_match(self):
        assert numeric_tolerance_match("100", "100", tolerance=0.5) == pytest.approx(1.0)

    def test_within_tolerance(self):
        # 99 vs 100 → 1% relative error, within 50% tolerance
        assert numeric_tolerance_match("100", "99", tolerance=0.5) == pytest.approx(1.0)

    def test_outside_tolerance(self):
        # 200 vs 100 → 100% relative error, outside 50% tolerance
        assert numeric_tolerance_match("100", "200", tolerance=0.5) == pytest.approx(0.0)

    def test_non_numeric_returns_zero(self):
        # If we cannot extract a number, no tolerance match
        assert numeric_tolerance_match("yes", "no", tolerance=0.5) == pytest.approx(0.0)

    def test_negative_numbers_close(self):
        # -558 vs -560 → very small relative error
        assert numeric_tolerance_match("-558", "-560", tolerance=0.5) == pytest.approx(1.0)


class TestEdgeCases:
    """Edge cases that have caused real bugs in the past."""

    def test_f1_with_whitespace(self):
        assert f1_score("  yes  ", "yes") == pytest.approx(1.0)

    def test_format_match_with_extra_spaces(self):
        assert format_match("  yes  ", "no") == pytest.approx(1.0)

    def test_normalize_idempotent(self):
        # normalize(normalize(x)) == normalize(x)
        once = normalize_answer("The Quick Fox!")
        twice = normalize_answer(once)
        assert once == twice
