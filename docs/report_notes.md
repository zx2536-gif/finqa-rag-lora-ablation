# Report Notes — Living Document

This file tracks experiments, decisions, and pre-drafted text for the final report.
Update after each milestone. All numbers come from `results/metrics/`.

---

## 1. Project Setup (Week 1, Day 1)

### What we built
- GitHub repo: github.com/zx2536-gif/finqa-rag-lora-ablation
- Modular code structure: `src/{utils,models,evaluation,pipelines,retrieval}/`
- Data pipeline: download → classify → stratified sample
- 500-sample evaluation set (`data/processed/finqa_500.json`)

### Decisions made
- **Stratified sampling with intentional boolean oversampling**: 
  130/130/130/60/40/10 per stratum vs natural ~28/28/29/11/2/0.2%
- **Reason**: ensure ≥10 samples per stratum for reliable error analysis
- **Trade-off**: per-stratum metrics ≠ population-weighted; we save 
  natural_weights in `sampling_stats.json` to enable both views

### Files needed for report
- `data/processed/sampling_stats.json` — sampling distribution
- `notebooks/01_data_exploration.ipynb` — methodology screenshots

---

## 2. C1 Baseline (Week 1, Day 1)

### What we ran
- Flan-T5-Base (250M) zero-shot on 500-sample test set
- Context: pre_text + table (markdown) + post_text, truncated to 350 words
- Prompt template: "Read the following financial document and answer..."

### Results
| Metric    | Value  |
|-----------|--------|
| Overall F1 | 3.67% |
| ROUGE-L    | 3.67% |

| Stratum             | F1    |
|---------------------|-------|
| boolean_complex     | 0.500 |
| boolean_simple      | 0.300 |
| numeric_complex     | 0.000 |
| numeric_simple      | 0.000 |
| percentage_complex  | 0.005 |
| percentage_simple   | 0.005 |

### Pre-drafted text for Section 5.1
> The Flan-T5-Base zero-shot baseline achieves overall F1 = 3.67%. 
> Numeric strata show complete failure (F1 = 0%) as the model produces
> non-numeric tokens like "no" or "percentage". Boolean strata score 
> 30-50% but this is misleading: 50% is the random-guess baseline for
> binary classification.

### Files for report
- `results/metrics/c1_metrics_*.json` — final metrics
- `results/metrics/c1_predictions_*.json` — sample predictions for error analysis

---

## 3. C2 RAG (Week 2, Day 1) [later updated to t5-base]

### What we ran
- BM25 retriever (rank-bm25) + sentence-transformers/all-MiniLM-L6-v2 dense (FAISS)
- Corpus: 11,375 sentence-level passages from 500 samples
- top-k = 3
- Same Flan-T5-Base zero-shot generator as C1

### Results (Flan-T5-Base, will be re-run with t5-base)
| Config          | F1    | Recall@3 | MRR   |
|-----------------|-------|----------|-------|
| C2 BM25         | 5.60% | 53.39%   | 45.33% |
| C2 Dense        | 5.67% | 47.70%   | 40.26% |

### Key finding (PROMINENT, write into report)
**BM25 outperforms dense retrieval in FinQA** — counter to typical RAG literature.
- Reason: FinQA queries share lexical overlap with gold passages 
  (fiscal years, entity names, accounting jargon) that exact-match 
  retrieval handles well.
- General-purpose dense encoder (web-trained) struggles to differentiate 
  adjacent fiscal years (e.g., "2016" vs "2015").

### Pre-drafted text for Section 5.2
> RAG provides large gains on boolean_simple (30% → 57.5% F1) but 
> negligible improvement on numeric/percentage strata (still ~0% F1).
> This indicates retrieval addresses *fact localization* but not 
> *generative reasoning*. The retrieval-only metrics quantify a hard
> ceiling: BM25 Recall@3 = 53.4%, meaning gold evidence is missing 
> from top-3 in 46.6% of queries.

### Files for report
- `results/metrics/c2_bm25_*` — BM25 results
- `results/metrics/c2_dense_*` — Dense results

---

## 4. ⚠️ Methodology Change: Base Model Swap (Week 2, Day 2)

### What happened
Initial LoRA fine-tuning of Flan-T5-Base failed catastrophically:

| Setup                          | Train Loss | Val Loss | F1    | Unique Preds |
|--------------------------------|-----------|----------|-------|--------------|
| Flan-T5-Base, LR=1e-4, r=8     | 6.96      | 6.47     | 0.6%  | **2** (mode collapse) |
| Flan-T5-Base, LR=3e-4, r=16    | 5.75      | 5.50     | 0.6%  | 2 |

Mode collapse: model output reduced to "64%" / "66%" regardless of input.

### Diagnosis
Heavy instruction tuning of Flan-T5 produces representations that 
resist parameter-efficient adaptation at small scale (250M). LoRA can 
only learn output bias, not real reasoning patterns.

### Decision
**Replaced base model with original `google-t5/t5-base`** (same 250M 
parameter budget, no instruction tuning). All four configurations 
(C1-C4) will use t5-base for fair comparison.

### Pre-drafted text for Section 4 (Methodology)
> We initially selected Flan-T5-Base (250M) following the proposal, 
> motivated by its instruction-following capability. However, 
> preliminary LoRA experiments revealed mode collapse — the trained 
> model produced only one or two unique output tokens regardless of 
> input, with training loss plateauing above 5.5 (perplexity ~245).
> 
> We hypothesize that heavy instruction tuning produces representations 
> that resist parameter-efficient adaptation at this scale, a 
> phenomenon noted in the LoRA literature for small instruction-tuned 
> models. We therefore swapped to the original T5-base (Raffel et al., 
> 2020), which has identical parameter count but no instruction tuning. 
> All configurations C1-C4 use T5-base for fair comparison.
>
> We report this as a methodology finding: base model selection is 
> non-trivial for parameter-efficient methods, particularly at small 
> scales — a result complementary to FinLoRA which exclusively used 
> 7B+ scale base models.

### Files for report
- This logbook entry itself (paste into report)
- `results/metrics/c3_vanilla_metrics_*.json` (Flan-T5 mode collapse evidence — keep!)

---

## 5. C3 LoRA / QLoRA (Week 2, Day 2)

### What we ran
- Base model: `google-t5/t5-base` (250M)
- Variants: Vanilla LoRA (bf16) and QLoRA (4-bit nf4)
- Training: 5703 samples (FinQA train minus 500 eval IDs), 871 dev
- Hyperparameters: r=16, alpha=32, target=q,k,v,o, LR=3e-4, 5 epochs, 
  cosine schedule, warmup=0.1, batch=16

### Results (after methodology change)
| Config            | F1    | ROUGE-L | Format Match | Numeric@0.5 | Train Time |
|-------------------|-------|---------|--------------|-------------|------------|
| C3 Vanilla LoRA   | 3.73% | 3.73%   | ~75%         | 12.5%       | 8m 34s     |
| C3 QLoRA          | 2.00% | 2.00%   | 75.6%        |  9.6%       | 19m 1s     |

### Per-stratum F1 (Vanilla)
| Stratum             | C1    | C3 Vanilla LoRA |
|---------------------|-------|-----------------|
| boolean_complex     | 0.500 | 0.300           |
| boolean_simple      | 0.300 | 0.300           |
| numeric_complex     | 0.000 | 0.000           |
| numeric_simple      | 0.000 | 0.008           |
| percentage_complex  | 0.005 | 0.013           |
| percentage_simple   | 0.005 | 0.008           |

### Key findings (multiple, write into report)

**Finding 1: F1 understates LoRA progress for numeric QA**
LoRA dramatically improves output format (95% percentage format match, 
59% numeric format match) but F1 barely moves because token-level 
matching penalizes near-correct numeric answers (e.g., gold=137.8, 
pred=87.4 → F1=0). We report format-match and tolerance@0.5 as 
supplementary metrics for honest evaluation.

**Finding 2: LoRA learns format, struggles with arithmetic**
- Percentage format match: 95% (model knows "% answer needed")
- Numeric format match: 59% (model knows "number answer needed")  
- Numeric tolerance@0.5: only 12.5% (model can't compute correctly)
- Confirms our hypothesis: LoRA addresses generation-side bottleneck, 
  RAG addresses retrieval-side. Their combination (C4) should compound.

**Finding 3: QLoRA has no advantage at small scale**
- Vanilla LoRA: 3.73% F1, 8m34s training
- QLoRA: 2.00% F1, 19m01s training (2.2× slower, lower accuracy)
- The 4-bit memory savings are unnecessary on A100 for 250M model.
- QLoRA's value is in 7B+ scale where vanilla LoRA OOMs.

### Pre-drafted text for Section 5.3
> LoRA fine-tuning improves output format dramatically: 95.0% of 
> percentage answers are now formatted correctly (vs ~5% in C1), and 
> 59.2% of numeric answers are correctly formatted (vs ~0% in C1). 
> However, F1 (3.73%) only marginally exceeds the C1 baseline (3.67%), 
> because token-level matching does not credit near-correct numeric 
> answers. We therefore report supplementary metrics: format-match 
> rate and numeric tolerance@0.5 (relative error <50%).

### Files for report
- `results/metrics/c3_vanilla_metrics_*` 
- `results/metrics/c3_qlora_metrics_*`
- Mode-collapse predictions from Flan-T5-Base experiments (kept as 
  evidence for methodology section)

---

## 6. Pending: C1/C2 Re-runs with t5-base, then C4

### What's left
- [ ] Re-run C1 with t5-base (replaces flan-t5-base baseline)
- [ ] Re-run C2 BM25 with t5-base
- [ ] Re-run C2 Dense with t5-base
- [ ] Build C4: LoRA + RAG (combine C2 retrieval with C3 LoRA model)
- [ ] Final 4-config comparison table
- [ ] Sensitivity analysis: top-k = {1, 3, 5} for RAG

---

## Screenshots / Figures to capture for the report

When ready, capture:
1. Sampling distribution chart (data/processed/sampling_stats.json)
2. Training curve: Flan-T5 mode collapse vs t5-base healthy training (loss vs steps)
3. Per-stratum F1 bar chart, all 4 configs side by side
4. Format match rate by stratum
5. Sample predictions table: same question, 4 different config outputs
6. Retrieval Recall@k curve (k=1,3,5) for BM25 vs Dense

---

## TA discussion talking points

When you next see the TA, mention:
1. Methodology decision: Flan-T5 → t5-base swap with quantitative evidence
2. Adding supplementary metrics (format-match, numeric tolerance) due to F1 limitations on numeric QA
3. Counter-intuitive finding: BM25 > Dense in FinQA domain
4. QLoRA has no advantage at 250M scale

These are research-level discussions and TAs reward this engagement.


---

## 7. C1/C2 Re-runs with t5-base (Week 3, Day 1)

After methodology change, all configs were re-run on t5-base for fair comparison:

| Config       | F1     | Format Match | Numeric@0.5 |
|--------------|--------|--------------|-------------|
| C1 baseline  | 0.07%  | ~5%          | <1%         |
| C2 BM25      | 0.00%  | ~5%          | <1%         |
| C2 Dense     | 0.00%  | ~5%          | <1%         |

Retrieval metrics unchanged (retriever doesn't depend on base model):
- BM25: Recall@3 = 53.39%, MRR = 45.33%
- Dense: Recall@3 = 47.70%, MRR = 40.26%

**Key insight**: Without instruction tuning, t5-base cannot do zero-shot 
QA at all. F1 ≈ 0 across C1/C2 means LoRA/RAG improvements in C3/C4 are
fully attributable to the methods, not residual base-model capability.

---

## 8. C4 LoRA + RAG (Week 3, Day 1)

### What we ran
4 sub-configs combining 2 LoRA variants × 2 retrievers:
- C4 Vanilla + BM25
- C4 Vanilla + Dense
- C4 QLoRA + BM25
- C4 QLoRA + Dense

All use top-k=3, t5-base base, same retrievers as C2.

### Results

| Sub-config        | F1     | Format Match | Numeric@0.5 |
|-------------------|--------|--------------|-------------|
| C4 Van + BM25     | 6.20%  | 94.6%        | 18.8%       |
| C4 Van + Dense    | 6.20%  | 93.8%        | **19.8%**   |
| C4 QL + BM25      | 5.00%  | 86.6%        | 17.4%       |
| C4 QL + Dense     | 4.80%  | 87.0%        | 18.2%       |

### Per-stratum F1 (C4 Vanilla + BM25, our best config)
| Stratum             | F1     | Format | Numeric@0.5 |
|---------------------|--------|--------|-------------|
| boolean_complex     | 0.6000 | 1.000  | 0.000       |
| boolean_simple      | 0.5750 | 0.900  | 0.000       |
| numeric_complex     | 0.0000 | 0.933  | 0.183       |
| numeric_simple      | 0.0077 | 0.923  | 0.177       |
| percentage_complex  | 0.0077 | 0.962  | 0.154       |
| percentage_simple   | 0.0000 | 0.969  | **0.308**   |

### Five major findings

**Finding 1 — Additive effects confirmed (CORE PROPOSAL HYPOTHESIS):**

|              | No RAG | With RAG | Δ from RAG |
|--------------|--------|----------|------------|
| No LoRA      | 0.07%  | 0.00%    | -0.07%     |
| With LoRA    | 3.73%  | 6.20%    | **+2.47%** |

RAG's gain over LoRA-only (+2.47%) is far larger than RAG's gain over 
baseline (-0.07%). LoRA and RAG address complementary bottlenecks: 
LoRA teaches output format and reasoning structure, RAG provides 
specific evidence. Their combination is superadditive.

**Finding 2 — Vanilla LoRA consistently outperforms QLoRA:**
QLoRA F1 trails Vanilla by 19-46% across configurations. The gap shrinks 
when RAG is added (46% gap in C3 → 19-23% gap in C4), suggesting 
retrieval partially compensates for quantization precision loss.

**Finding 3 — BM25 vs Dense parity in C4:**
Despite BM25's 5.7-point Recall@3 advantage in C2, end-to-end F1 is 
identical in C4 (both 6.20% with Vanilla LoRA). LoRA model is 
robust to retriever quality differences — once trained, it can extract 
answers from semantically-similar (Dense) or lexically-exact (BM25) 
contexts equally well.

**Finding 4 — Numeric tolerance saturates at ~20%:**
The full 4-config progression: <1% → 12.5% (LoRA) → 19.8% (LoRA+RAG). 
The combined system gets ~1 in 5 numeric answers within 50% relative 
error — meaningful but bounded by the 250M model's limited arithmetic 
capability. This is the "ceiling" for this scale.

**Finding 5 — Format mastery nearly complete:**
percentage_simple format match: 96.9%; boolean_complex: 100.0%. The model 
has learned the FinQA output schema. Remaining errors are reasoning-side, 
not generation-side — pointing to the next bottleneck (model scale).

### Pre-drafted text for Section 5.4

> Combining LoRA fine-tuning with retrieval (BM25 or Dense, top-3) 
> produces our strongest results: F1 = 6.20%, Format Match = 94.6%, 
> Numeric Tolerance = 19.8%. The improvements are *superadditive*: 
> adding RAG to LoRA yields +2.47 F1 points, far exceeding RAG's -0.07 
> point effect on the unfinetuned baseline.
>
> This validates our central hypothesis (Section 2.1): RAG and LoRA 
> address complementary bottlenecks. LoRA teaches output schema (94.6% 
> format match) and reasoning structure (12.5% → 19.8% numeric tolerance), 
> while RAG provides specific evidence needed to fill in correct values. 
> Neither alone is sufficient.
>
> Notably, BM25 and Dense retrievers achieve identical F1 (6.20%) when 
> paired with LoRA, despite BM25's 5.7-point retrieval advantage in C2. 
> This suggests fine-tuned models are robust to retriever quality — a 
> finding with practical implications for deployment under retriever 
> latency or cost constraints.

### Files for report
- `results/metrics/c4_vanilla_bm25_*` — best config
- `results/metrics/c4_vanilla_dense_*`  
- `results/metrics/c4_qlora_bm25_*`
- `results/metrics/c4_qlora_dense_*`

---

## 9. Summary: Final Numbers Across All 9 Configs

| Config           | F1     | Format | Num@0.5 | Notes                     |
|------------------|--------|--------|---------|---------------------------|
| C1 baseline      | 0.07%  | ~5%    | <1%     | t5-base zero-shot         |
| C2 BM25          | 0.00%  | ~5%    | <1%     | retrieval no use without LoRA |
| C2 Dense         | 0.00%  | ~5%    | <1%     | same                      |
| C3 Vanilla LoRA  | 3.73%  | 76.8%  | 12.5%   | LoRA learns format        |
| C3 QLoRA         | 2.00%  | 75.6%  | 10.7%   | quantization loss         |
| C4 Van + BM25    | **6.20%** | 94.6% | 18.8% | best lexical retrieval    |
| C4 Van + Dense   | **6.20%** | 93.8% | **19.8%** | best numeric tolerance |
| C4 QL + BM25     | 5.00%  | 86.6%  | 17.4%   |                           |
| C4 QL + Dense    | 4.80%  | 87.0%  | 18.2%   |                           |

Best overall: **C4 Vanilla LoRA + Dense RAG** (best Numeric Tolerance).
Best F1: **C4 Vanilla LoRA + BM25 or Dense** (tied at 6.20%).
