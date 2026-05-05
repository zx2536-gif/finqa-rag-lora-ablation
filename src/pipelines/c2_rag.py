"""
C2: RAG-only pipeline (no fine-tuning).

Retrieves top-k passages for each question using BM25 or dense retrieval,
then generates an answer with Flan-T5-Base zero-shot. Reports both
end-to-end QA metrics (F1, ROUGE-L) and retrieval-only metrics (Recall@k, MRR).

Usage:
    python -m src.pipelines.c2_rag --retriever bm25
    python -m src.pipelines.c2_rag --retriever dense --top_k 3
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
from src.models.baseline import FlanT5Baseline
from src.evaluation.metrics import evaluate_predictions


def build_rag_prompt(question: str, retrieved_passages, max_context_words: int = 350) -> str:
    """Build a prompt using retrieved passages as context.

    Args:
        question: the question string
        retrieved_passages: list of (passage_dict, score) tuples
        max_context_words: cap to fit Flan-T5's 512-token limit
    """
    # Concatenate retrieved passages with separators
    context_parts = [p['text'] for p, _ in retrieved_passages]
    context = " ".join(context_parts)

    # Truncate to fit token budget
    words = context.split()
    if len(words) > max_context_words:
        context = " ".join(words[:max_context_words]) + " [...]"

    return (
        "Read the following financial evidence and answer the question. "
        "Give a short, direct answer (a number, percentage, or yes/no).\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def run_c2(data_path: str, out_dir: str,
           retriever_type: str = 'bm25',
           top_k: int = 3,
           model_name: str = 'google/flan-t5-base',
           n_samples: int = None,
           batch_size: int = 16):
    # === Step 1: Load data ===
    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        samples = json.load(f)
    if n_samples:
        samples = samples[:n_samples]
    print(f"  Loaded {len(samples)} samples")

    # === Step 2: Build corpus & queries ===
    print(f"\nBuilding passage corpus...")
    passages = build_passage_corpus(samples)
    queries = build_query_set(samples)
    print(f"  {len(passages)} passages, {len(queries)} queries")
    n_with_gold = sum(1 for q in queries if q['gold_passage_ids'])
    print(f"  {n_with_gold}/{len(queries)} queries have gold evidence "
          f"({n_with_gold/len(queries)*100:.1f}%)")

    # === Step 3: Build retriever ===
    print(f"\nBuilding {retriever_type} retriever...")
    if retriever_type == 'bm25':
        retriever = BM25Retriever(passages)
    elif retriever_type == 'dense':
        retriever = DenseRetriever(passages)
    else:
        raise ValueError(f"Unknown retriever: {retriever_type}")

    # === Step 4: Retrieve top-k for all queries ===
    print(f"\nRetrieving top-{top_k} for {len(queries)} queries...")
    question_strs = [q['question'] for q in queries]
    retrieval_results = retriever.batch_retrieve(question_strs, top_k=top_k)

    # === Step 5: Retrieval-only evaluation ===
    print("\nEvaluating retrieval...")
    retrieval_metrics = evaluate_retrieval(retrieval_results, queries,
                                           k_values=[1, 3, 5])

    if 'error' not in retrieval_metrics:
        print(f"  Recall@1: {retrieval_metrics['overall']['recall@1']:.4f}")
        print(f"  Recall@3: {retrieval_metrics['overall']['recall@3']:.4f}")
        print(f"  MRR:      {retrieval_metrics['overall']['mrr']:.4f}")
    else:
        print(f"  {retrieval_metrics['error']}")

    # === Step 6: Build RAG prompts and generate ===
    print(f"\nBuilding RAG prompts and loading model...")
    prompts = [build_rag_prompt(q['question'], r)
               for q, r in zip(queries, retrieval_results)]

    model = FlanT5Baseline(model_name=model_name)
    print(f"\nGenerating predictions (batch_size={batch_size})...")
    predictions = model.predict_batch(prompts, batch_size=batch_size)

    # === Step 7: End-to-end QA evaluation ===
    golds = [q['gold_answer'] for q in queries]
    strata = [q['stratum'] for q in queries]

    print("\nEvaluating QA...")
    qa_metrics = evaluate_predictions(predictions, golds, strata)

    print(f"\n=== C2 RAG Results ({retriever_type}, top-{top_k}) ===")
    print(f"Overall ({qa_metrics['overall']['n_samples']} samples):")
    print(f"  F1:      {qa_metrics['overall']['f1']:.4f}")
    print(f"  ROUGE-L: {qa_metrics['overall']['rouge_l']:.4f}")
    print(f"\nPer-stratum:")
    for st in sorted(qa_metrics['by_stratum'].keys()):
        m = qa_metrics['by_stratum'][st]
        print(f"  {st:25s} n={m['n_samples']:3d}  "
              f"F1={m['f1']:.4f}  ROUGE-L={m['rouge_l']:.4f}")

    # === Step 8: Save outputs ===
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'c2_{retriever_type}_k{top_k}'

    metrics_path = os.path.join(out_dir, f'{tag}_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': f'C2_RAG_{retriever_type}',
            'retriever': retriever_type,
            'top_k': top_k,
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
                {'passage_id': p['passage_id'], 'text': p['text'], 'score': s}
                for p, s in retrieval_results[i]
            ],
        } for i in range(len(queries))]
        json.dump(records, f, indent=2)
    print(f"Saved predictions to {preds_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/finqa_500.json')
    parser.add_argument('--out_dir', default='results/metrics')
    parser.add_argument('--retriever', choices=['bm25', 'dense'], default='bm25')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--model', default='google/flan-t5-base')
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    run_c2(args.data, args.out_dir, args.retriever, args.top_k,
           args.model, args.n_samples, args.batch_size)
