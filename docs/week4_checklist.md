# Week 4 Action Items (Final Sprint)

## Total Estimated Time: 8-12 hours, distributed over 4-6 days

---

## Priority 1: Visualizations (3-5 hours) — DO THIS FIRST

These are referenced throughout the Section 5 pre-drafted text. Without 
figures, the report reads incompletely.

- [ ] **Figure 1**: Stratified sampling distribution (natural % vs sampled %)
  - Source: `data/processed/sampling_stats.json`
  - Type: grouped bar chart
- [ ] **Figure 2**: Mode collapse evidence (Flan-T5-Base vs t5-base loss curves)
  - Source: training logs from earlier runs
  - Type: line chart, 2 series
  - This figure visually justifies the methodology change
- [ ] **Figure 3**: 9-config F1 bar chart (overall)
  - Source: all metrics JSON files
  - Type: horizontal bar chart, 9 bars
- [ ] **Figure 4**: Per-stratum F1 heatmap (6 strata × 9 configs)
  - Source: all metrics JSON
  - Type: heatmap with annotations
- [ ] **Figure 5**: Format match progression (C1 → C2 → C3 → C4)
  - Source: all metrics
  - Type: stacked or grouped bar showing the staircase progression
- [ ] **Figure 6**: BM25 vs Dense Recall@k comparison
  - Source: c2 metrics retrieval section
  - Type: line chart, 2 series, x=k, y=Recall@k
- [ ] **Figure 7**: Sample predictions table (qualitative)
  - Source: prediction JSONs
  - Type: 4-row LaTeX table showing same Q across configs
- [ ] **Figure 8**: Format vs Numeric tolerance gap (showing the gap shrinks with C4)
  - Source: format_match and numeric_tolerance@0.5 across configs
  - Type: dual-axis or grouped bar

## Priority 2: Gradio Demo (2-3 hours) — Rubric 20%

- [ ] Create `demo/app.py` with side-by-side comparison
- [ ] Load 4 configs (C1, C3 vanilla, C4 van+bm25, C4 van+dense)
- [ ] Input field: question + (optional) financial document
- [ ] Output: 4 columns showing each config's prediction
- [ ] 5 preset example questions covering all 6 strata
- [ ] Test deployment locally before demo day

## Priority 3: Final Report Writing (3-4 hours) — Rubric 30%

Most of Section 4-5 is pre-drafted in `docs/report_notes.md`. Remaining:

- [ ] Section 1: Abstract (200 words, write after everything else)
- [ ] Section 2: Introduction (1 page; expand from proposal section 1)
- [ ] Section 3: Related Work (1.5 pages; expand from proposal references)
- [ ] Section 4: Methodology — paste from logbook + add architecture diagram
- [ ] Section 5: Experiments — paste from logbook + insert figures
- [ ] Section 6: Error Analysis — pick 5-10 representative failure cases
  - [ ] Find samples where C1 = C4 = wrong (model limit)
  - [ ] Find samples where C3 wrong, C4 right (RAG benefit)
  - [ ] Find samples where C1 wrong, C3 right (LoRA benefit)
- [ ] Section 7: Limitations
  - Small base model (250M) limits arithmetic capability
  - Single-domain (financial) — generalization untested
  - Closed-corpus retrieval — open-domain RAG would be harder
  - Boolean strata small (n=10 for boolean_complex) — variance high
- [ ] Section 8: Conclusion — restate findings 1-6 from logbook
- [ ] References — collect from proposal + add new ones for findings

## Priority 4: Presentation Slides (2-3 hours) — Rubric 30%

Suggested 12-slide structure:
1. Title + team
2. Problem & motivation (financial QA challenges)
3. Research questions (the 3 from proposal)
4. Approach: 2x2 ablation diagram
5. Methodology: data, base model, hyperparameters
6. Methodology finding: Flan-T5 → t5-base swap (quantitative)
7. Results overview: 9-config F1 bar chart
8. Finding 1: Additive effects (C4 > C3 + C2)
9. Finding 2: BM25 > Dense in finance (counter-intuitive)
10. Finding 3: Format vs reasoning bottleneck (C3 vs C4 numeric tolerance)
11. Limitations & future work
12. Demo + Q&A

## Priority 5: GitHub README polish (30 min) — Rubric 10%

- [ ] Update README with final results table
- [ ] Add quick-start instructions: `python -m src.pipelines.c4_lora_rag --variant vanilla --retriever bm25`
- [ ] Add "Reproducibility" section with seed, hardware, training time
- [ ] Add link to demo (after Gradio is deployed)

---

## Risk Mitigation

If running short on time, prioritize:
1. **MUST HAVE**: Figures 3, 4, 5 (overall results visuals)
2. **MUST HAVE**: Sections 1, 4, 5, 6 in report (the empirical core)
3. **MUST HAVE**: Working Gradio demo
4. **MUST HAVE**: Slides 7, 8, 9, 10 (the findings)

If you run out of time, the optional cuts are:
- Sensitivity analysis (top-k=1,3,5)
- Figures 2, 7, 8 (nice-to-have)
- Error analysis depth (3-4 cases is OK if time-constrained)
