"""
Evaluation metrics for FinQA QA tasks.

Implements F1 and ROUGE-L for textual question answering, plus utilities
for aggregating results per stratum.
"""

import re
import string
from collections import Counter
from typing import List, Dict


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.
    
    Standard normalization from SQuAD and downstream QA benchmarks.
    """
    s = (s or "").lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer.
    
    Returns 0.0 if either is empty (after normalization).
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)  # 1.0 if both empty, else 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    """Longest Common Subsequence length, used by ROUGE-L."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F-measure between prediction and gold."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    lcs = lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(predictions: List[str], golds: List[str], 
                         strata: List[str] = None) -> Dict:
    """Compute aggregate F1, ROUGE-L, format match, and numeric tolerance.
    
    Returns Dict with 'overall' and optional 'by_stratum' breakdowns.
    """
    assert len(predictions) == len(golds), \
        f"Mismatched lengths: {len(predictions)} preds vs {len(golds)} golds"
    
    f1s = [f1_score(p, g) for p, g in zip(predictions, golds)]
    rouges = [rouge_l(p, g) for p, g in zip(predictions, golds)]
    fmts = [format_match(p, g, strata[i] if strata else None) 
            for i, (p, g) in enumerate(zip(predictions, golds))]
    nums = [numeric_tolerance_match(p, g) for p, g in zip(predictions, golds)]
    
    results = {
        'overall': {
            'n_samples': len(predictions),
            'f1': sum(f1s) / len(f1s),
            'rouge_l': sum(rouges) / len(rouges),
            'format_match': sum(fmts) / len(fmts),
            'numeric_tolerance@0.5': sum(nums) / len(nums),
        }
    }
    
    if strata is not None:
        assert len(strata) == len(predictions)
        by_stratum = {}
        for stratum in set(strata):
            indices = [i for i, s in enumerate(strata) if s == stratum]
            by_stratum[stratum] = {
                'n_samples': len(indices),
                'f1': sum(f1s[i] for i in indices) / len(indices),
                'rouge_l': sum(rouges[i] for i in indices) / len(indices),
                'format_match': sum(fmts[i] for i in indices) / len(indices),
                'numeric_tolerance@0.5': sum(nums[i] for i in indices) / len(indices),
            }
        results['by_stratum'] = by_stratum
    
    return results


def classify_answer_format(s: str) -> str:
    """Classify answer string by output format: percentage / numeric / boolean / other."""
    import re
    s = (s or '').strip().lower()
    if s in ('yes', 'no', 'true', 'false'):
        return 'boolean'
    if '%' in s:
        return 'percentage'
    if re.match(r'^-?[\d,.]+\s*(million|billion|thousand)?$', s.replace(' ', '')):
        return 'numeric'
    return 'other'


def format_match(prediction: str, gold: str, stratum: str = None) -> float:
    """1.0 if prediction format matches gold's format (or stratum's expected format)."""
    pred_fmt = classify_answer_format(prediction)
    if stratum:
        # Use the stratum prefix as the expected format (e.g. "percentage_simple" -> "percentage")
        expected = stratum.split('_')[0]
        return float(pred_fmt == expected)
    else:
        gold_fmt = classify_answer_format(gold)
        return float(pred_fmt == gold_fmt)


def extract_number(s: str):
    """Extract first signed decimal from a string. Returns None if no number."""
    import re
    m = re.search(r'-?\d+\.?\d*', str(s))
    return float(m.group()) if m else None


def numeric_tolerance_match(prediction: str, gold: str, tolerance: float = 0.5) -> float:
    """1.0 if extracted numbers from prediction and gold are within `tolerance` relative error.
    
    Returns 0.0 if either side has no extractable number, or if relative error >= tolerance.
    Special case: both numbers ~0 -> 1.0.
    """
    g = extract_number(gold)
    p = extract_number(prediction)
    if g is None or p is None:
        return 0.0
    if abs(g) <= 1e-3:
        return float(abs(p) <= 1e-3)
    return float(abs(p - g) / abs(g) < tolerance)
