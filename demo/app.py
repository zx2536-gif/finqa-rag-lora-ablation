"""
FinQA 4-Configuration Comparison Demo

A Gradio interface that runs the same financial QA query through all four
configurations of our ablation study (C1/C2/C3/C4) and displays predictions
side-by-side. Includes 5 preset examples covering different answer types
and difficulty levels.

Run:
    cd /content/drive/MyDrive/finqa-rag-lora-ablation
    python demo/app.py
"""

import json
import os
import sys
import warnings

import gradio as gr
import torch

warnings.filterwarnings('ignore')

# Make src importable (works for both `python app.py` and exec/Jupyter)
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _here = '/content/drive/MyDrive/finqa-rag-lora-ablation/demo'
sys.path.insert(0, os.path.dirname(_here))

from src.utils.data_utils import build_prompt, build_context, table_to_markdown
from src.retrieval.corpus import build_passage_corpus
from src.retrieval.bm25_retriever import BM25Retriever
from src.pipelines.c2_rag import build_rag_prompt
from src.pipelines.c3_lora import FlanT5LoRA

# ----------------------------------------------------------------------------
# Global model/retriever state (loaded once at startup)
# ----------------------------------------------------------------------------
print("=" * 70)
print("Initializing FinQA Demo...")
print("=" * 70)

# Load demo samples (used both for examples and for building the BM25 corpus)
DEMO_DATA_PATH = 'data/processed/finqa_500.json'
print(f"\n[1/4] Loading samples from {DEMO_DATA_PATH}...")
with open(DEMO_DATA_PATH) as f:
    DEMO_SAMPLES = json.load(f)
print(f"      {len(DEMO_SAMPLES)} samples loaded.")

# Build BM25 corpus (needed for C2 and C4 RAG configs)
print(f"\n[2/4] Building BM25 retriever over {len(DEMO_SAMPLES)} samples...")
DEMO_PASSAGES = build_passage_corpus(DEMO_SAMPLES)
BM25 = BM25Retriever(DEMO_PASSAGES)
print(f"      Corpus: {len(DEMO_PASSAGES)} passages indexed.")

# Load C1 baseline (no LoRA, no fine-tuning)
print(f"\n[3/4] Loading C1 baseline (t5-base zero-shot)...")
from src.models.baseline import FlanT5Baseline
C1_MODEL = FlanT5Baseline(model_name='google-t5/t5-base')

# Load C3/C4 model (Vanilla LoRA, used for both)
print(f"\n[4/4] Loading C3/C4 model (Vanilla LoRA adapter)...")
ADAPTER_DIR = 'checkpoints/c3_vanilla/final'
LORA_MODEL = FlanT5LoRA(adapter_dir=ADAPTER_DIR,
                        base_model='google-t5/t5-base',
                        variant='vanilla')

print("\n" + "=" * 70)
print("✅ All models loaded. Demo ready.")
print("=" * 70)


# ----------------------------------------------------------------------------
# Inference functions for each config
# ----------------------------------------------------------------------------
def run_c1(question, document):
    """C1: zero-shot baseline (no LoRA, no RAG)."""
    sample = {'pre_text': [document], 'post_text': [], 'table': [],
              'qa': {'question': question}}
    prompt = build_prompt(sample)
    return C1_MODEL.predict_batch([prompt], batch_size=1)[0]


def run_c2(question, document):
    """C2: BM25 RAG, no fine-tuning."""
    # Retrieve top-3 from the global corpus
    retrieved = BM25.retrieve(question, top_k=3)
    prompt = build_rag_prompt(question, retrieved)
    pred = C1_MODEL.predict_batch([prompt], batch_size=1)[0]
    return pred, retrieved


def run_c3(question, document):
    """C3: LoRA fine-tuning, no retrieval."""
    sample = {'pre_text': [document], 'post_text': [], 'table': [],
              'qa': {'question': question}}
    prompt = build_prompt(sample)
    return LORA_MODEL.predict_batch([prompt], batch_size=1)[0]


def run_c4(question, document):
    """C4: LoRA + BM25 RAG (best config)."""
    retrieved = BM25.retrieve(question, top_k=3)
    prompt = build_rag_prompt(question, retrieved)
    pred = LORA_MODEL.predict_batch([prompt], batch_size=1)[0]
    return pred, retrieved


def run_all(question, document):
    """Run all 4 configs and return formatted results."""
    if not question or not question.strip():
        return ("(no question provided)",) * 4 + ("",)
    
    document = document or ""
    
    try:
        c1_pred = run_c1(question, document)
        c2_pred, c2_retrieved = run_c2(question, document)
        c3_pred = run_c3(question, document)
        c4_pred, c4_retrieved = run_c4(question, document)
    except Exception as e:
        err = f"⚠️ Error: {str(e)[:200]}"
        return (err,) * 4 + ("",)
    
    # Format retrieved passages (showing C4's retrieval)
    retrieved_md = "### 🔎 Top-3 Retrieved Passages (used by C2 and C4)\n\n"
    for i, (p, score) in enumerate(c4_retrieved, 1):
        text_short = p['text'][:200] + ('...' if len(p['text']) > 200 else '')
        retrieved_md += f"**[{i}]** *(BM25 score: {score:.2f}, from `{p['sample_id']}`)*\n\n"
        retrieved_md += f"> {text_short}\n\n"
    
    return c1_pred, c2_pred, c3_pred, c4_pred, retrieved_md


# ----------------------------------------------------------------------------
# Pre-built examples covering different strata
# ----------------------------------------------------------------------------
def get_examples():
    """Pick representative examples from finqa_500 across strata."""
    examples_by_stratum = {}
    for s in DEMO_SAMPLES:
        st = s.get('_stratum', '')
        if st in examples_by_stratum:
            continue
        if st in ('boolean_simple', 'boolean_complex',
                  'percentage_simple', 'percentage_complex',
                  'numeric_simple'):
            doc = build_context(s)[:800]  # truncate for UI display
            examples_by_stratum[st] = [s['qa']['question'], doc, s['qa']['answer']]
        if len(examples_by_stratum) >= 5:
            break
    
    return list(examples_by_stratum.values())


EXAMPLES = get_examples()


# ----------------------------------------------------------------------------
# Gradio interface
# ----------------------------------------------------------------------------
def build_interface():
    custom_css = """
    .gradio-container {max-width: 1400px !important;}
    .config-output {min-height: 80px;}
    .config-card {padding: 12px; border-radius: 8px; margin: 4px;}
    """
    
    with gr.Blocks(title='FinQA 4-Config Comparison',
                   theme=gr.themes.Soft(primary_hue='blue'),
                   css=custom_css) as demo:
        
        gr.Markdown("""
        # 🏦 FinQA 4-Configuration Comparison
        
        ### Same question → 4 system configurations → side-by-side outputs
        
        Compare zero-shot baseline (C1), retrieval-only (C2), fine-tuning-only (C3),
        and combined (C4) on financial QA. **C4 is our best-performing system**
        from the ablation study (F1 = 6.20%, Format Match = 94.6%).
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                question_in = gr.Textbox(
                    label='Financial Question',
                    placeholder='e.g., what was the percentage change in net sales from 2014 to 2015?',
                    lines=2,
                )
                document_in = gr.Textbox(
                    label='Financial Document Context (optional)',
                    placeholder='Paste financial report text here. Leave blank to rely on retrieval (C2/C4).',
                    lines=6,
                )
                with gr.Row():
                    submit_btn = gr.Button('▶ Run All 4 Configs', variant='primary', scale=2)
                    clear_btn = gr.Button('Clear', scale=1)
            
            with gr.Column(scale=1):
                gr.Markdown("""
                ### How to use
                1. Enter a financial question
                2. (Optional) Paste a related financial document
                3. Click **Run All 4 Configs**
                4. Compare predictions across the 4 system designs
                
                ### Configurations
                - **C1**: t5-base zero-shot (no enhancement)
                - **C2**: + BM25 retrieval (top-3 passages)
                - **C3**: + LoRA fine-tuning on FinQA train
                - **C4**: + LoRA + RAG (★ best)
                
                ### Try the examples below
                Click any row to load a real FinQA test sample.
                """)
        
        gr.Markdown("---")
        gr.Markdown("## 📊 Predictions")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### C1: Baseline\n*zero-shot*")
                c1_out = gr.Textbox(label='', show_label=False, interactive=False,
                                     elem_classes=['config-output'])
            with gr.Column():
                gr.Markdown("### C2: RAG only\n*BM25 + zero-shot*")
                c2_out = gr.Textbox(label='', show_label=False, interactive=False,
                                     elem_classes=['config-output'])
            with gr.Column():
                gr.Markdown("### C3: LoRA only\n*fine-tuned, no retrieval*")
                c3_out = gr.Textbox(label='', show_label=False, interactive=False,
                                     elem_classes=['config-output'])
            with gr.Column():
                gr.Markdown("### C4: LoRA + RAG ★\n*best system*")
                c4_out = gr.Textbox(label='', show_label=False, interactive=False,
                                     elem_classes=['config-output'])
        
        retrieved_out = gr.Markdown('')
        
        gr.Markdown("---")
        gr.Markdown("## 📝 Try Real FinQA Examples")
        gr.Examples(
            examples=EXAMPLES,
            inputs=[question_in, document_in],
            label='Click an example to load it',
        )
        
        gr.Markdown("""
        ---
        ### About this demo
        Built for STAT GR5293 GenAI Course Project, Spring 2026. 
        Models: t5-base (250M) + LoRA (r=16, target=q,k,v,o, 1.56% trainable params).
        Retriever: BM25 over 11,375 sentence-level passages from FinQA test set.  
        See [github.com/zx2536-gif/finqa-rag-lora-ablation](https://github.com/zx2536-gif/finqa-rag-lora-ablation) for code & report.
        """)
        
        # Wire up callbacks
        submit_btn.click(
            fn=run_all,
            inputs=[question_in, document_in],
            outputs=[c1_out, c2_out, c3_out, c4_out, retrieved_out],
        )
        clear_btn.click(
            fn=lambda: ('', '', '', '', '', '', ''),
            outputs=[question_in, document_in, c1_out, c2_out, c3_out, c4_out, retrieved_out],
        )
    
    return demo


if __name__ == '__main__':
    demo = build_interface()
    demo.launch(share=True, server_port=7860, debug=False)
