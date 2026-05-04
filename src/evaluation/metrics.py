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
    """Compute aggregate F1 and ROUGE-L scores.
    
    Args:
        predictions: list of model output strings
        golds: list of ground truth strings  
        strata: optional list of stratum labels for per-stratum breakdown
    
    Returns:
        Dict with 'overall' metrics and optionally 'by_stratum'.
    """
    assert len(predictions) == len(golds), \
        f"Mismatched lengths: {len(predictions)} preds vs {len(golds)} golds"
    
    f1s = [f1_score(p, g) for p, g in zip(predictions, golds)]
    rouges = [rouge_l(p, g) for p, g in zip(predictions, golds)]
    
    results = {
        'overall': {
            'n_samples': len(predictions),
            'f1': sum(f1s) / len(f1s),
            'rouge_l': sum(rouges) / len(rouges),
        }
    }
    
    # Per-stratum breakdown
    if strata is not None:
        assert len(strata) == len(predictions)
        by_stratum = {}
        for stratum in set(strata):
            indices = [i for i, s in enumerate(strata) if s == stratum]
            by_stratum[stratum] = {
                'n_samples': len(indices),
                'f1': sum(f1s[i] for i in indices) / len(indices),
                'rouge_l': sum(rouges[i] for i in indices) / len(indices),
            }
        results['by_stratum'] = by_stratum
    
    return results
