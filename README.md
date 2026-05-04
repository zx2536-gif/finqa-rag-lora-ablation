# finqa-rag-lora-ablation
A controlled 2×2 ablation study of RAG and LoRA on FinQA — STAT GR5293 GenAI Course Project
# RAG vs. LoRA vs. RAG+LoRA: A Controlled Ablation Study for Financial QA

**STAT GR5293 GenAI Course Project (Spring 2026)**

A 2×2 ablation study comparing Retrieval-Augmented Generation (RAG) and 
Parameter-Efficient Fine-Tuning (LoRA / QLoRA) on the FinQA benchmark.

## Team
- Zezhou Xie (zx2536)
- Bochao Du (bd2779)

## Project Overview

This project investigates the independent and combined effects of retrieval 
augmentation and parameter-efficient fine-tuning on financial question answering. 
We evaluate four configurations on FinQA textual sub-tasks:

| Config | RAG | LoRA Fine-Tuning |
|--------|-----|------------------|
| C1: Baseline (zero-shot) | ✗ | ✗ |
| C2: RAG only | ✓ | ✗ |
| C3: LoRA only | ✗ | ✓ |
| C4: LoRA + RAG | ✓ | ✓ |

## Repository Structure
├── configs/         # Experiment configurations
├── data/            # Data preprocessing and samples
├── src/             # Core source code
├── notebooks/       # Colab notebooks (run in numerical order)
├── results/         # Experiment outputs and figures
├── demo/            # Gradio demo app
└── docs/            # Proposal, report, presentation
## Setup

### Requirements
- Python 3.10+
- CUDA-enabled GPU (tested on Colab Pro A100)

### Installation
```bash
git clone https://github.com/<username>/finqa-rag-lora-ablation.git
cd finqa-rag-lora-ablation
pip install -r requirements.txt
```

### Data
FinQA dataset is downloaded automatically by the preprocessing script:
```bash
python data/preprocess.py
```

## Reproducing Experiments

Run the notebooks in `notebooks/` in numerical order, or:

```bash
# C1: Baseline
python -m src.pipelines.c1_baseline --config configs/c1_baseline.yaml

# C2: RAG only
python -m src.pipelines.c2_rag --config configs/c2_rag_only.yaml

# C3: LoRA only
python -m src.pipelines.c3_lora --config configs/c3_lora_only.yaml

# C4: LoRA + RAG
python -m src.pipelines.c4_lora_rag --config configs/c4_lora_rag.yaml
```

## Demo

```bash
python demo/app.py
```

Then open `http://localhost:7860` in your browser.

## Results

[To be filled in after experiments — Week 2]

## References

See `docs/proposal.pdf` for full reference list.

## License

MIT
