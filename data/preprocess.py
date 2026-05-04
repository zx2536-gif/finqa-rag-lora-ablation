"""
FinQA Data Preprocessing Pipeline

Downloads the original FinQA dataset from GitHub, classifies samples by 
answer type (percentage/numeric/boolean) and complexity (simple/complex), 
performs stratified sampling with intentional oversampling of boolean 
questions for reliable error analysis.

Usage:
    python data/preprocess.py
    python data/preprocess.py --n_samples 500 --seed 42

Output:
    data/raw/{train,dev,test}.json     - original FinQA splits
    data/processed/finqa_500.json       - 500 stratified samples
    data/processed/sampling_stats.json  - sampling statistics for the report
    data/samples/finqa_10_demo.json     - 10-sample demo (committed to git)
"""

import argparse
import json
import os
import random
import re
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, List

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DATA_URLS = {
    'train': 'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json',
    'dev':   'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json',
    'test':  'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json',
}

# Target sample count per stratum (must sum to total N)
STRATUM_QUOTAS = {
    'percentage_simple':  130,
    'percentage_complex': 130,
    'numeric_simple':     130,
    'numeric_complex':     60,
    'boolean_simple':      40,
    'boolean_complex':     10,
}
TOTAL_SAMPLES = sum(STRATUM_QUOTAS.values())  # 500

BOOLEAN_OPS = {'greater', 'less', 'equal', 'smaller',
               'greater_or_equal', 'less_or_equal'}


# -----------------------------------------------------------------------------
# Classification functions
# -----------------------------------------------------------------------------

def classify_answer(answer: str, program: str) -> str:
    """Classify a sample by answer type: percentage / numeric / boolean / other."""
    answer = (answer or '').strip().lower()
    program = (program or '').strip()

    # 1. boolean: explicit yes/no, OR program ends with comparison op
    if answer in ('yes', 'no', 'true', 'false'):
        return 'boolean'
    if program:
        last_op = program.split(',')[-1].strip().split('(')[0].strip()
        if last_op in BOOLEAN_OPS:
            return 'boolean'

    # 2. percentage
    if '%' in answer:
        return 'percentage'

    # 3. numeric (allows $, comma, million/billion suffixes)
    cleaned = answer.replace('$', '').replace(',', '').replace(' ', '')
    cleaned = re.sub(r'(million|billion|thousand|m|bn|k)$', '', cleaned)
    try:
        float(cleaned)
        return 'numeric'
    except (ValueError, TypeError):
        return 'other'


def classify_complexity(program: str) -> str:
    """Simple if program has <=2 ops, complex otherwise."""
    if not program:
        return 'simple'
    n_ops = len([op for op in program.split(',') if op.strip()])
    return 'simple' if n_ops <= 2 else 'complex'


def get_stratum(sample: dict) -> str:
    """Return the stratum label for a sample, e.g. 'percentage_simple'."""
    qa = sample.get('qa', {}) or {}
    ans_type = classify_answer(qa.get('answer', ''), qa.get('program', ''))
    complexity = classify_complexity(qa.get('program', ''))
    return f"{ans_type}_{complexity}"


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def download_finqa(raw_dir: str = 'data/raw') -> Dict[str, list]:
    """Download FinQA splits from the official GitHub repo if not already present."""
    os.makedirs(raw_dir, exist_ok=True)
    data = {}
    for split, url in DATA_URLS.items():
        path = os.path.join(raw_dir, f'{split}.json')
        if not os.path.exists(path):
            print(f"Downloading {split}.json ...")
            urllib.request.urlretrieve(url, path)
        with open(path, 'r') as f:
            data[split] = json.load(f)
        print(f"  {split}: {len(data[split])} samples")
    return data


# -----------------------------------------------------------------------------
# Stratified sampling
# -----------------------------------------------------------------------------

def stratified_sample(samples: List[dict], quotas: Dict[str, int],
                      seed: int = 42) -> Dict[str, list]:
    """
    Bucket samples by stratum, then randomly draw `quotas[stratum]` from each.
    Returns dict mapping stratum -> list of sampled records.
    Raises if any stratum has fewer than the requested quota.
    """
    rng = random.Random(seed)

    buckets = defaultdict(list)
    for s in samples:
        stratum = get_stratum(s)
        if stratum in quotas:
            buckets[stratum].append(s)

    sampled = {}
    for stratum, quota in quotas.items():
        pool = buckets.get(stratum, [])
        if len(pool) < quota:
            raise ValueError(
                f"Stratum {stratum!r} has only {len(pool)} samples but "
                f"{quota} were requested. Adjust STRATUM_QUOTAS."
            )
        sampled[stratum] = rng.sample(pool, quota)
        print(f"  {stratum:25s}: drew {quota:3d} from pool of {len(pool):4d}")

    return sampled


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--raw_dir', default='data/raw')
    parser.add_argument('--out_dir', default='data/processed')
    parser.add_argument('--demo_dir', default='data/samples')
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Download FinQA from GitHub")
    print("=" * 60)
    finqa = download_finqa(args.raw_dir)

    print("\n" + "=" * 60)
    print("Step 2: Compute full distribution on train split")
    print("=" * 60)
    full_strata = Counter(get_stratum(s) for s in finqa['train'])
    total = sum(full_strata.values())
    natural_weights = {}
    for stratum in STRATUM_QUOTAS:
        n = full_strata.get(stratum, 0)
        natural_weights[stratum] = n / total if total > 0 else 0
        print(f"  {stratum:25s}: {n:5d} ({natural_weights[stratum]*100:.1f}%)")

    print("\n" + "=" * 60)
    print(f"Step 3: Stratified sample {TOTAL_SAMPLES} from train")
    print("=" * 60)
    sampled_by_stratum = stratified_sample(finqa['train'], STRATUM_QUOTAS,
                                            seed=args.seed)

    # Flatten to single list, attach stratum metadata for downstream use
    flat_samples = []
    for stratum, items in sampled_by_stratum.items():
        for item in items:
            item['_stratum'] = stratum
            flat_samples.append(item)

    print(f"\n  Total sampled: {len(flat_samples)}")

    print("\n" + "=" * 60)
    print("Step 4: Save outputs")
    print("=" * 60)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.demo_dir, exist_ok=True)

    # Main processed file (NOT committed, in .gitignore)
    out_path = os.path.join(args.out_dir, f'finqa_{TOTAL_SAMPLES}.json')
    with open(out_path, 'w') as f:
        json.dump(flat_samples, f, indent=2)
    print(f"  Saved: {out_path}")

    # Sampling statistics (committed, useful for the report)
    stats = {
        'total_samples': TOTAL_SAMPLES,
        'random_seed': args.seed,
        'stratum_quotas': STRATUM_QUOTAS,
        'natural_weights': natural_weights,
        'note': 'Boolean strata are intentionally oversampled '
                '(~10% sampled vs ~3% natural) for reliable error analysis. '
                'Use natural_weights to recompute population-weighted metrics.'
    }
    stats_path = os.path.join(args.out_dir, 'sampling_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # Demo file with 10 samples (committed for TA reproducibility)
    demo = flat_samples[:10]
    demo_path = os.path.join(args.demo_dir, 'finqa_10_demo.json')
    with open(demo_path, 'w') as f:
        json.dump(demo, f, indent=2)
    print(f"  Saved: {demo_path}")

    print("\n✅ Done.")


if __name__ == '__main__':
    main()
