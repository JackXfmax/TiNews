# TiNews — A Diagnostic Benchmark for Tibetan News Understanding

> **One-line:** Standard metrics (PPL, BLEU) are blind to a class of *systematic
> structural* failures that general LLMs make on Tibetan news. TiNews exposes them
> with a syllable-level orthography diagnostic (**ISR**) and an Information-Bottleneck /
> KL decomposition of compression behaviour.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TiNews is a three-task benchmark for Tibetan (བོད་ཡིག) news understanding, built as a
**diagnostic testbed** (not a leaderboard). Its central claim: general-purpose LLMs
suffer *orthography-level* and *compression-level* failures on Tibetan that are
invisible to perplexity and BLEU, and a dedicated model + lightweight adaptation
(TiLamb-7B + LoRA) closes the gap to human-level legality.

---

## Tasks

| Task | Description | n | Primary metric |
|---|---|---|---|
| **T1 — Classification** | 7-class Tibetan news topic classification | 1049 | macro-F1, confusion matrix |
| **T2 — Continuation** | Sentence-level continuation (motivation hook + recovery) | 200 | bits/syll, ISR |
| **T3 — Summarization + Keywords** | Abstractive summarization (BodSUM-6000) + keyword generation (Ti-SUM) | 594 / 992 | ISR, chrF, ROUGE-L, BLEU-4 |

## The ISR metric (center of the benchmark)

**Illegal Syllable Rate (ISR)** = fraction of generated syllables that violate Tibetan
syllable orthography (the *དབྱིབས་སྒྲིག* / syllable-construction constraints). It is the
benchmark's central diagnostic because:

- It catches failures PPL/BLEU cannot see (e.g. a fluent-looking summary that is
  orthographically broken, or a model that *falls out of Tibetan* into Chinese).
- It is validated against **3.6M human syllables** (ISR ≈ 0.4 %–1.0 %), so it does
  not over-penalize legitimate Tibetan.
- Human reference summaries score ISR ≈ **0.004**; this is the anchor for "human-level".

> ⚠️ **Cross-tokenizer warning (red line).** PPL is *not* comparable across tokenizers.
> The only comparable quantity is **bits per Tibetan syllable** (bridged in
> `data/results_ppl_*.json`). Never compare raw PPL between TiLamb and Qwen/Llama.

## Compression invariance (Theorem 3.2′)

The original monotonic-rate hypothesis was **falsified** by a 7-bracket length sweep
(Qwen ISR *rises* with compression — the wrong direction). It is replaced by a
two-layer decomposition:

- a **semantic / IB layer** governed by the information budget `L_min(i) = i / h_eff`;
- an **orthography / glyph-grounding layer** governed by a model-specific deficit
  `γ(M)` (determined *only* by vocabulary + pretraining weights, sitting *below* the
  token level, orthogonal to semantics).

**Theorem 3.2′ (compression invariance):** under (A1) token-MLE training and
(A2) stationary glyph-grounding position, `ISR(M,L) = γ(M) + O_p(L^{-1/2})`, so
compression can *neither induce nor repair* orthography failure. (A2) is an
*empirically verified* claim (fixed-budget ISR is position-stationary), **not** a
first-principle proof — do not write "proved" in the text.

## Repository structure

```
TiNews/
├── README.md                 # this file
├── LICENSE                   # MIT
├── CITATION.cff              # citation metadata (fill in author / repo URL)
├── MANIFEST.md               # full data index + speed-lookup of every number
├── data/                     # all computed metrics + system outputs (see data/README.md)
│   ├── results_*.json        # point estimates & aggregates (one file per experiment)
│   ├── raw_hyps/             # per-item ref/hyp for summarization (抽查用)
│   ├── human_eval_200.*      # our 200-sample human-evaluation set (with rater scores)
│   └── ppl_peritem.*         # per-item PPL for the bits/syll bridge
├── fig_data/                 # the 8 figures / appendix tables (paper-ready JSON)
│   └── pmax_raw.npy          # raw p_max array for Fig. C3
├── scripts/                  # diagnostic & significance code (reproducible method)
│   ├── bootstrap_task3.py        # 12 paired bootstrap CIs (摘要)
│   ├── bootstrap_gen_metrics.py  # paired bootstrap CIs (续写)
│   ├── human_trunc_control.py    # human-summary truncation control (rules out truncation artifact)
│   ├── positional_isr.py         # fixed-budget ISR position-stationarity (breaks circularity)
│   └── task3_len_sweep.py        # 7-bracket length sweep
└── docs/                     # paper material (outline, theory, ledger, figure report)
    ├── figure_data_report.md     # ★ the 8-figure integrated report (copy numbers to LaTeX)
    ├── results_ledger.md         # ★ authoritative 60KB results ledger
    ├── TiNews_contribution_v6.md # theory (Def 1–3 + Theorem 3.2′ + Corollary 3.3)
    ├── TiNews_FULL_PAPER_MATERIAL.md  # single-file paper material (all numbers)
    ├── TiNews_human_eval_final.md     # human-eval final numbers (实测)
    ├── TiNews_outline_rev4.md    # latest paper outline
    ├── TiNews_limitations_draft.md    # Limitations draft (10 points, anchored to data)
    ├── TiNews_task3_spec.md      # T3 spec
    ├── TiNews_g1_draft.tex       # drafted §G1 fragment
    └── archive/                  # superseded drafts (rev1–3, v5, expected human-eval)
```

## Quick start

```bash
# all numbers live in data/ (JSON) and fig_data/ (JSON) — no training needed to cite
python -c "import json; d=json.load(open('data/results_g1_tilamb_lora.json')); print(d)"
```

To reproduce a figure, load the matching `fig_data/fig_*.json` (field paths are listed
in `docs/figure_data_report.md` §9). To re-run a significance test, see `scripts/`.

## Data provenance (important)

- **Our contributions** (shipped here): all metric JSON, `fig_data/`, `scripts/`,
  `docs/`, and the `human_eval_200` set — released under **MIT** (code) /
  **CC-BY-4.0** (data & docs).
- **Third-party raw text** (BodSUM test split, Ti-SUM parsed split, evaluation
  contexts) is **not redistributed** in this repo. Obtain it from the original
  sources under their respective licenses and place it locally (see `.gitignore`).
  The benchmark's *aggregated statistics* over those corpora (length distributions,
  cluster/keyword coverage) are included in `fig_data/` and need no raw text.

## Citation

```bibtex
@software{TiNews2026,
  title  = {TiNews: A Diagnostic Benchmark for Tibetan News Understanding
            under Syllable Orthography Constraints},
  author = {[AUTHOR NAME]},
  year   = {2026},
  url    = {https://github.com/[GITHUB USERNAME]/TiNews},
  license = {MIT}
}
```

See `CITATION.cff` for the machine-readable form. **Before citing in the paper,
replace the `[AUTHOR NAME]` / `[GITHUB USERNAME]` placeholders** (and add ORCID).

## License

Code: **MIT**. Data & documentation: **CC-BY-4.0**. See `LICENSE`.

---

*Status: all experimental numbers are cross-validated (local recompute vs. server
re-run, bit-identical). Self-falsification of the original rate hypothesis is kept
on record as an asset, not a defect (see `MANIFEST.md` §4 red lines).*
