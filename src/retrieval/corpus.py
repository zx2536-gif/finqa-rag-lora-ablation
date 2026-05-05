"""
Corpus builder for RAG pipelines on FinQA.

Splits each FinQA sample's pre_text and post_text into individual sentence-level
passages, attaching metadata (sample_id, source, position) needed for retrieval
evaluation against FinQA's gold text_retrieved annotations.

FinQA's text_retrieved field uses continuous indexing: text_N refers to
pre_text[N] when N < len(pre_text), else post_text[N - len(pre_text)].
"""

import json
from typing import List, Dict, Any


def build_passage_corpus(samples: List[Dict[str, Any]]) -> List[Dict]:
    """Convert FinQA samples into a flat list of sentence-level passages.

    Each pre_text/post_text sentence becomes one passage, with a unique
    passage_id constructed from sample_id, source, and position. The schema
    matches the gold-evidence indexing used in build_query_set.

    Returns:
        List of passage dicts, each with:
            - passage_id: unique global id (e.g., "ABMD/2015/page_93.pdf-1__pre_text__3")
            - text:       the passage content
            - sample_id:  original FinQA sample id
            - source:     'pre_text' or 'post_text'
            - position:   index within source
    """
    passages = []
    for s_idx, sample in enumerate(samples):
        sample_id = sample.get('id', f'sample_{s_idx}')

        for src_field in ('pre_text', 'post_text'):
            for pos, sent in enumerate(sample.get(src_field, [])):
                sent = (sent or '').strip()
                if not sent or sent == '.':
                    continue
                passages.append({
                    'passage_id': f"{sample_id}__{src_field}__{pos}",
                    'text': sent,
                    'sample_id': sample_id,
                    'source': src_field,
                    'position': pos,
                })
    return passages


def build_query_set(samples: List[Dict[str, Any]]) -> List[Dict]:
    """Extract one query per sample with its gold-evidence passage_ids.

    Decodes FinQA's text_retrieved field into passage_ids that match
    those produced by build_passage_corpus.

    Returns:
        List of query dicts, each with:
            - query_id, question, gold_answer
            - gold_passage_ids: list of passage_ids for the gold evidence
            - stratum: for downstream per-stratum analysis
    """
    queries = []
    for s_idx, sample in enumerate(samples):
        sample_id = sample.get('id', f'sample_{s_idx}')
        qa = sample.get('qa', {})

        pre_text = sample.get('pre_text', [])
        post_text = sample.get('post_text', [])
        n_pre = len(pre_text)

        # Decode each {'ind': 'text_N'} into a passage_id
        gold_passage_ids = []
        for item in sample.get('text_retrieved', []):
            ind_str = item.get('ind', '') if isinstance(item, dict) else ''
            if not ind_str.startswith('text_'):
                continue
            try:
                idx = int(ind_str.split('_', 1)[1])
            except ValueError:
                continue

            # Continuous indexing: pre_text first, then post_text
            if idx < n_pre:
                # Skip if the indexed sentence is empty/dot (matches passage_corpus filter)
                sent = (pre_text[idx] or '').strip()
                if sent and sent != '.':
                    gold_passage_ids.append(
                        f"{sample_id}__pre_text__{idx}"
                    )
            else:
                post_idx = idx - n_pre
                if post_idx < len(post_text):
                    sent = (post_text[post_idx] or '').strip()
                    if sent and sent != '.':
                        gold_passage_ids.append(
                            f"{sample_id}__post_text__{post_idx}"
                        )

        queries.append({
            'query_id': sample_id,
            'question': qa.get('question', ''),
            'gold_answer': str(qa.get('answer', '')).strip(),
            'gold_passage_ids': gold_passage_ids,
            'stratum': sample.get('_stratum', 'unknown'),
        })
    return queries


def save_corpus(passages: List[Dict], queries: List[Dict], out_dir: str):
    """Save corpus and queries to JSON for downstream pipelines."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/passages.json', 'w') as f:
        json.dump(passages, f, indent=2)
    with open(f'{out_dir}/queries.json', 'w') as f:
        json.dump(queries, f, indent=2)
    print(f"  Saved {len(passages)} passages to {out_dir}/passages.json")
    print(f"  Saved {len(queries)} queries to {out_dir}/queries.json")


if __name__ == '__main__':
    # Quick smoke test
    with open('data/processed/finqa_500.json') as f:
        samples = json.load(f)
    passages = build_passage_corpus(samples)
    queries = build_query_set(samples)

    print(f"Total passages: {len(passages)}")
    print(f"Total queries:  {len(queries)}")
    queries_with_gold = sum(1 for q in queries if q['gold_passage_ids'])
    print(f"Queries with at least 1 gold passage: {queries_with_gold} "
          f"({queries_with_gold/len(queries)*100:.1f}%)")

    # Show one example to verify decoding
    for q in queries:
        if q['gold_passage_ids']:
            print(f"\nExample query:")
            print(f"  question: {q['question'][:100]}")
            print(f"  gold_passage_ids: {q['gold_passage_ids']}")
            break

    save_corpus(passages, queries, 'data/processed/corpus')
