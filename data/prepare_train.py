"""
Prepare training data for C3 LoRA fine-tuning.

Filters out the 500 samples used for evaluation (finqa_500.json) from the
original FinQA training set, then formats remaining samples as 
input-target pairs for sequence-to-sequence training.

Usage:
    python data/prepare_train.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_utils import build_context


MAX_CONTEXT_WORDS = 350  # match C1/C2 truncation


def format_input(sample: dict) -> str:
    """Format a FinQA sample into a Flan-T5 input string."""
    context = build_context(sample)
    words = context.split()
    if len(words) > MAX_CONTEXT_WORDS:
        context = " ".join(words[:MAX_CONTEXT_WORDS]) + " [...]"
    
    question = sample.get('qa', {}).get('question', '')
    return (
        "Read the following financial document and answer the question. "
        "Give a short, direct answer (a number, percentage, or yes/no).\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def format_target(sample: dict) -> str:
    """Format the gold answer string for training."""
    return str(sample.get('qa', {}).get('answer', '')).strip()


def main():
    # Load splits
    raw_dir = 'data/raw'
    with open(f'{raw_dir}/train.json') as f:
        train_full = json.load(f)
    with open(f'{raw_dir}/dev.json') as f:
        dev_full = json.load(f)
    print(f"Original splits: train={len(train_full)}, dev={len(dev_full)}")
    
    # Load eval set (500 samples) and extract their IDs to exclude
    with open('data/processed/finqa_500.json') as f:
        eval_samples = json.load(f)
    eval_ids = {s['id'] for s in eval_samples}
    print(f"Eval set IDs to exclude: {len(eval_ids)}")
    
    # Filter train: keep only samples NOT in eval set
    train_filtered = [s for s in train_full if s.get('id') not in eval_ids]
    print(f"Filtered train: {len(train_filtered)} "
          f"(excluded {len(train_full) - len(train_filtered)})")
    
    # Format as input-target pairs, drop samples with empty answer
    def to_pairs(samples):
        pairs = []
        for s in samples:
            target = format_target(s)
            if not target:  # skip samples with empty answer
                continue
            pairs.append({
                'id': s.get('id', ''),
                'input': format_input(s),
                'target': target,
            })
        return pairs
    
    train_pairs = to_pairs(train_filtered)
    dev_pairs = to_pairs(dev_full)
    print(f"Final pairs: train={len(train_pairs)}, dev={len(dev_pairs)}")
    
    # Save
    out_dir = 'data/processed/training'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/train.json', 'w') as f:
        json.dump(train_pairs, f, indent=2)
    with open(f'{out_dir}/dev.json', 'w') as f:
        json.dump(dev_pairs, f, indent=2)
    
    print(f"\nSaved to {out_dir}/")
    
    # Show a sample
    print("\n=== Example training pair ===")
    print("INPUT (first 500 chars):")
    print(train_pairs[0]['input'][:500])
    print("\nTARGET:")
    print(repr(train_pairs[0]['target']))


if __name__ == '__main__':
    main()
