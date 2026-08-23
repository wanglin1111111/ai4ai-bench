# AI4AI-Bench: Auditing Recursive Self-Improvement in LLM Agents

**SIAS (Self-Improvement Audit Score)** — a quantitative framework for auditing whether agent self-improvement (runtime harness editing) is genuine versus overfitting or degenerative.

## Key Findings (Real Data)

Auditing the **27 published harness patches** from the [Harness-R1](https://github.com/DeepExperience/Harness-R1) held-out generalization study (1,270 held-out tasks; WebShop 490 + ALFWorld 490 + DBBench 290; 3 seeds):

| Editor | Valid | Pooled held-out delta | Rescued/Regressed | Destruction rate |
|---|---|---|---|---|
| Harness-R1 (RL, trained 9B) | 9/9 | **+8.92 ± 1.50 pp** | 543/203 | 27% |
| Qwen3.5-397B | 8/9 | **−4.33 ± 2.52 pp** | 129/294 | **70%** |
| DeepSeek-V4-Pro | 6/9 | **−0.42 ± 3.58 pp** | 191/207 | 52% |

- Two frontier editors are **net-negative out-of-batch**; one real patch rescued 0 failures while destroying 71 successes (textbook MyopicFix).
- A subset-audit gate with a **50-task budget reaches 87% ± 6% decision accuracy** and recovers **+251 net tasks (+19.8pp)** vs. indiscriminate acceptance.
- Patch complexity (hook count) does NOT predict quality (single-hook: +44.7 for Harness-R1 vs. −47.0 for Qwen3.5-397B); the rescued-vs-regressed destruction structure is the informative signal.

## Repository Contents

| Path | Description |
|---|---|
| `paper/main.pdf` | Paper (NeurIPS-style, 6 pages) |
| `paper/main.tex` + `references.bib` + `neurips_2026.sty` | LaTeX sources |
| `paper/figures_*.png` | Figures |
| `src/real_data_audit.py` | Retrospective audit of the 27 real patches (main script) |
| `src/ai4ai_bench_evaluator.py` | SIAS evaluator: score, 6 degradation patterns, recursive protocol |
| `src/ai4ai_bench_experiment.py` | Simulation experiment framework |
| `src/run_ai4ai_bench_experiments.py` | One-command reproduction pipeline |
| `results/real_data_audit_results.json` | Audit results on real data |
| `results/table1/2/3_*.csv` | Real-data result tables |

## Data Source (public, Apache 2.0)

All real-data numbers derive from the official records released by the Harness-R1 authors:
- Repo: https://github.com/DeepExperience/Harness-R1
- Path: `examples/heldout_generalization/results.json`

## Reproduce

```bash
# Real-data audit (point at a local clone or download of the JSON above)
export HARNESS_R1_RESULTS=/path/to/Harness-R1/examples/heldout_generalization/results.json
python src/real_data_audit.py

# Simulation pipeline (generates figures + tables)
python src/run_ai4ai_bench_experiments.py
```

## Honesty Notes

- Real-data evaluation is a **retrospective audit** of published patches; prospective live deployment is future work.
- Simulation results in the paper are explicitly labeled "Simulation".
- The `neurips_2026.sty` here is the official 2024 style file renamed for this project; replace with the current year's official style for camera-ready.

## License

Code: MIT. The audited data belongs to its upstream authors (Apache 2.0).
