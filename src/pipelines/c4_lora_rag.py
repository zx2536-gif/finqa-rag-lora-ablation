"""
C4: LoRA + RAG combined pipeline.

Combines a trained LoRA adapter (from C3) with a retriever (from C2) to
produce the full ablation: parameter-efficient fine-tuning AND retrieval
augmentation. This is the central comparison of the proposal.

Usage:
    python -m src.pipelines.c4_lora_rag --variant vanilla --retriever bm25
    python -m src.pipelines.c4_lora_rag --variant qlora --retriever dense
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.retrieval.corpus import build_passage_corpus, build_query_set
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.eval_retrieval import evaluate_retrieval
from src.pipelines.c3_lora import FlanT5LoRA  # reuse the LoRA inference wrapper
from src.pipelines.c2_rag import build_rag_prompt  # reuse the RAG prompt builder
from src.evaluation.metrics import evaluate_predictions


def run_c4(adapter_dir: str,
           variant: str,
           retriever_type: str,
           data_path: str = 'data/processed/finqa_500.json',
           out_dir: str = 'results/metrics',
           top_k: int = 3,
           model_name: str = 'google-t5/t5-base',
           batch_size: int = 16):
    
    # === Step 1: Load data ===
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples")

    # === Step 2: Build corpus & queries ===
    print(f"\nBuilding passage corpus...")
    passages = build_passage_corpus(samples)
    queries = build_query_set(samples)
    print(f"  {len(passages)} passages, {len(queries)} queries")

    # === Step 3: Build retriever ===
    print(f"\nBuilding {retriever_type} retriever...")
    if retriever_type == 'bm25':
        retriever = BM25Retriever(passages)
    elif retriever_type == 'dense':
        retriever = DenseRetriever(passages)
    else:
        raise ValueError(f"Unknown retriever: {retriever_type}")

    # === Step 4: Retrieve top-k ===
    print(f"\nRetrieving top-{top_k} for {len(queries)} queries...")
    question_strs = [q['question'] for q in queries]
    retrieval_results = retriever.batch_retrieve(question_strs, top_k=top_k)

    # === Step 5: Retrieval-only metrics (sanity check, should match C2) ===
    print("\nEvaluating retrieval...")
    retrieval_metrics = evaluate_retrieval(retrieval_results, queries,
                                           k_values=[1, 3, 5])
    if 'error' not in retrieval_metrics:
        print(f"  Recall@1: {retrieval_metrics['overall']['recall@1']:.4f}")
        print(f"  Recall@3: {retrieval_metrics['overall']['recall@3']:.4f}")
        print(f"  MRR:      {retrieval_metrics['overall']['mrr']:.4f}")

    # === Step 6: Build RAG prompts ===
    print("\nBuilding RAG prompts...")
    prompts = [build_rag_prompt(q['question'], r)
               for q, r in zip(queries, retrieval_results)]

    # === Step 7: Load LoRA model and generate ===
    model = FlanT5LoRA(adapter_dir, base_model=model_name, variant=variant)
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    predictions = model.predict_batch(prompts, batch_size=batch_size)

    # === Step 8: QA evaluation ===
    golds = [q['gold_answer'] for q in queries]
    strata = [q['stratum'] for q in queries]

    print("\nEvaluating QA...")
    qa_metrics = evaluate_predictions(predictions, golds, strata)

    print(f"\n=== C4 {variant.upper()} LoRA + {retriever_type.upper()} RAG Results ===")
    print(f"Overall ({qa_metrics['overall']['n_samples']} samples):")
    print(f"  F1:                    {qa_metrics['overall']['f1']:.4f}")
    print(f"  ROUGE-L:               {qa_metrics['overall']['rouge_l']:.4f}")
    print(f"  Format match:          {qa_metrics['overall']['format_match']:.4f}")
    print(f"  Numeric tolerance@0.5: {qa_metrics['overall']['numeric_tolerance@0.5']:.4f}")
    print(f"\nPer-stratum F1:")
    for st in sorted(qa_metrics['by_stratum'].keys()):
        m = qa_metrics['by_stratum'][st]
        print(f"  {st:25s} n={m['n_samples']:3d}  "
              f"F1={m['f1']:.4f}  Fmt={m['format_match']:.4f}  "
              f"Num={m['numeric_tolerance@0.5']:.4f}")

    # === Step 9: Save outputs ===
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'c4_{variant}_{retriever_type}_k{top_k}'

    metrics_path = os.path.join(out_dir, f'{tag}_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': f'C4_LoRA_{variant}_RAG_{retriever_type}',
            'variant': variant,
            'retriever': retriever_type,
            'top_k': top_k,
            'adapter_dir': adapter_dir,
            'model': model_name,
            'data_path': data_path,
            'n_samples': len(samples),
            'timestamp': timestamp,
            'qa_metrics': qa_metrics,
            'retrieval_metrics': retrieval_metrics,
        }, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    preds_path = os.path.join(out_dir, f'{tag}_predictions_{timestamp}.json')
    with open(preds_path, 'w') as f:
        records = [{
            'id': queries[i]['query_id'],
            'stratum': strata[i],
            'question': queries[i]['question'],
            'gold': golds[i],
            'prediction': predictions[i],
            'retrieved_passages': [
                {'passage_id': p['passage_id'], 'text': p['text'][:200]}
                for p, _ in retrieval_results[i]
            ],
        } for i in range(len(queries))]
        json.dump(records, f, indent=2)
    print(f"Saved predictions to {preds_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['vanilla', 'qlora'], required=True)
    parser.add_argument('--retriever', choices=['bm25', 'dense'], required=True)
    parser.add_argument('--adapter_dir', default=None,
                        help='Default: checkpoints/c3_{variant}/final')
    parser.add_argument('--data', default='data/processed/finqa_500.json')
    parser.add_argument('--out_dir', default='results/metrics')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if args.adapter_dir is None:
        args.adapter_dir = f'/content/drive/MyDrive/finqa-rag-lora-ablation/checkpoints/c3_{args.variant}/final'

    run_c4(args.adapter_dir, args.variant, args.retriever, args.data,
           args.out_dir, args.top_k, batch_size=args.batch_size)
