"""
Retrieval-only evaluation metrics: Recall@k and MRR.

These metrics evaluate the retriever in isolation, before any LLM generation.
Useful for diagnosing whether QA failures are caused by retrieval or generation.
"""

from typing import List, Dict, Tuple


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    """1.0 if any gold passage is in the top-k retrieved, else 0.0."""
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return float(any(g in top_k for g in gold_ids))


def reciprocal_rank(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    """1 / (rank of first gold passage), or 0 if no gold in retrieved."""
    if not gold_ids:
        return 0.0
    gold_set = set(gold_ids)
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_set:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(retrieval_results: List[List[Tuple[Dict, float]]],
                       queries: List[Dict],
                       k_values: List[int] = (1, 3, 5)) -> Dict:
    """Compute retrieval metrics over the full query set.

    Args:
        retrieval_results: per query, list of (passage_dict, score) tuples (ranked)
        queries: aligned list of query dicts with 'gold_passage_ids'
        k_values: which k's to evaluate Recall@k for

    Returns:
        Dict with overall metrics + per-stratum breakdown.
    """
    assert len(retrieval_results) == len(queries)

    # Filter to queries that have at least one gold passage (else metric is undefined)
    valid_indices = [i for i, q in enumerate(queries) if q['gold_passage_ids']]

    if not valid_indices:
        return {'error': 'No queries have gold_passage_ids; cannot compute metrics.'}

    metrics_per_query = []
    for i in valid_indices:
        retrieved_ids = [p['passage_id'] for p, _ in retrieval_results[i]]
        gold_ids = queries[i]['gold_passage_ids']
        m = {'stratum': queries[i]['stratum'],
             'mrr': reciprocal_rank(retrieved_ids, gold_ids)}
        for k in k_values:
            m[f'recall@{k}'] = recall_at_k(retrieved_ids, gold_ids, k)
        metrics_per_query.append(m)

    # Aggregate overall
    overall = {'n_queries_with_gold': len(valid_indices)}
    for k in k_values:
        key = f'recall@{k}'
        overall[key] = sum(m[key] for m in metrics_per_query) / len(metrics_per_query)
    overall['mrr'] = sum(m['mrr'] for m in metrics_per_query) / len(metrics_per_query)

    # Per-stratum breakdown
    by_stratum = {}
    strata = sorted(set(m['stratum'] for m in metrics_per_query))
    for st in strata:
        sub = [m for m in metrics_per_query if m['stratum'] == st]
        by_stratum[st] = {'n': len(sub),
                          'mrr': sum(m['mrr'] for m in sub) / len(sub)}
        for k in k_values:
            key = f'recall@{k}'
            by_stratum[st][key] = sum(m[key] for m in sub) / len(sub)

    return {'overall': overall, 'by_stratum': by_stratum}
