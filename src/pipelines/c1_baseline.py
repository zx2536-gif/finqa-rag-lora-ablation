"""
C1: Zero-shot baseline pipeline.

Loads stratified FinQA samples, generates predictions using Flan-T5-Base
with no retrieval and no fine-tuning, computes F1 and ROUGE-L metrics.

Usage:
    python -m src.pipelines.c1_baseline
    python -m src.pipelines.c1_baseline --n_samples 50  # quick test
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Allow running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.utils.data_utils import build_prompt, get_gold_answer
from src.models.baseline import FlanT5Baseline
from src.evaluation.metrics import evaluate_predictions


def run_c1(data_path: str, out_dir: str, model_name: str = "google-t5/t5-base",
           n_samples: int = None, batch_size: int = 8):
    """Run the C1 zero-shot baseline pipeline end-to-end."""
    
    # === Step 1: Load data ===
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        samples = json.load(f)
    if n_samples:
        samples = samples[:n_samples]
        print(f"  Subsampled to {n_samples} for quick testing")
    print(f"  Loaded {len(samples)} samples")
    
    # === Step 2: Build prompts and gold answers ===
    print("\nBuilding prompts...")
    prompts = [build_prompt(s) for s in samples]
    golds = [get_gold_answer(s) for s in samples]
    strata = [s['_stratum'] for s in samples]
    
    # === Step 3: Load model and run inference ===
    model = FlanT5Baseline(model_name=model_name)
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    predictions = model.predict_batch(prompts, batch_size=batch_size)
    
    # === Step 4: Evaluate ===
    print("\nEvaluating...")
    results = evaluate_predictions(predictions, golds, strata)
    
    print(f"\n=== C1 Baseline Results ===")
    print(f"Overall ({results['overall']['n_samples']} samples):")
    print(f"  F1:      {results['overall']['f1']:.4f}")
    print(f"  ROUGE-L: {results['overall']['rouge_l']:.4f}")
    
    print(f"\nPer-stratum:")
    for stratum, m in sorted(results['by_stratum'].items()):
        print(f"  {stratum:25s}  n={m['n_samples']:3d}  "
              f"F1={m['f1']:.4f}  ROUGE-L={m['rouge_l']:.4f}")
    
    # === Step 5: Save outputs ===
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Metrics file (committed to git via results/metrics/)
    metrics_path = os.path.join(out_dir, f'c1_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': 'C1_zero_shot_baseline',
            'model': model_name,
            'data_path': data_path,
            'n_samples': len(samples),
            'timestamp': timestamp,
            'metrics': results,
        }, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Predictions file (for error analysis)
    preds_path = os.path.join(out_dir, f'c1_predictions_{timestamp}.json')
    with open(preds_path, 'w') as f:
        records = [{
            'id': samples[i].get('id', f'sample_{i}'),
            'stratum': strata[i],
            'question': samples[i]['qa']['question'],
            'gold': golds[i],
            'prediction': predictions[i],
        } for i in range(len(samples))]
        json.dump(records, f, indent=2)
    print(f"Saved predictions to {preds_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/finqa_500.json')
    parser.add_argument('--out_dir', default='results/metrics')
    parser.add_argument('--model', default='google-t5/t5-base')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Optional cap for quick testing (e.g., 50)')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    run_c1(args.data, args.out_dir, args.model, args.n_samples, args.batch_size)
