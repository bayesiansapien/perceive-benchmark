# PERCEIVE: A Self-Expanding Benchmark for Psychophysics-driven Elicitation of Routing Cost-Efficiency In Vision-Language Evaluation

**NeurIPS 2026 Datasets and Benchmarks Track**

PERCEIVE is a 4,801-sample document-image QA benchmark for cost-aware VLM routing. Each sample carries psychophysical complexity annotations (Visual Dependency Score, Reasoning Depth Score, Spatial Extent Score) derived from a two-model probe oracle (GPT-5.4-mini and Gemini-2.5-Flash-Lite), a difficulty tier (Easy/Medium/Hard), and a routing label identifying the cheapest model-budget configuration that answers it correctly.

Routing labels are derived via a QUEST-style adaptive cascade at **60.7% cost reduction** with **100% ground-truth label agreement** across 7 commercial VLMs at up to 4 reasoning-budget levels (24 valid configurations).

**Dataset:** https://huggingface.co/datasets/quantiphi-routing/perceive-benchmark

---

## Repository structure

```
perceive/
├── configs/
│   ├── datasets.yaml          # 16 source dataset definitions, sample budgets, metrics
│   ├── model_pool.yaml        # 7 VLMs x up to 4 reasoning budgets (24 valid configs), pricing
│   └── annotation_rubric.yaml # Probe prompt rubric for VDS/RDS/SES elicitation
├── data/
│   ├── croissant.json                    # NeurIPS Croissant metadata (RAI fields, checksums)
│   ├── routing_labels/
│   │   └── routing_labels.jsonl          # 4,801 cheapest-correct routing labels
│   ├── anchor_set/                       # 1,500-sample anchor set IDs and selection report
│   ├── validation_set/                   # 750-sample validation set IDs
│   ├── sample_evidence/                  # 4 annotated example images (Appendix F)
│   ├── phase4_results/                   # Router training and baseline comparison results
│   ├── imc_external_validation/          # IMC new-model validation (Exp 1.1a: AUC 0.833-0.873)
│   ├── imc_dataset_holdout_report.json   # IMC cross-domain holdout (Exp 1.1b: AUC 0.60)
│   ├── oracle_gap_decomposition.json     # Oracle gap analysis results
│   ├── judge_sensitivity_*.json          # Judge sensitivity experiment results
│   ├── cascade_mf_results/               # Cascade-MF baseline results
│   └── phase2_sensitivity_report.json    # Probe sensitivity experiment results
├── src/
│   ├── ingestion/             # Dataset adapters for all 16 source datasets
│   ├── sampling/              # Stratified submodular facility-location sampler
│   ├── model_eval/            # VLM evaluation harness (API + GPU adapters)
│   ├── scoring/               # Per-metric scorers: ANLS, exact match, IoU, F1, etc.
│   ├── anchor_set/            # Anchor set selection (submodular facility location)
│   └── matrix_completion/     # Inductive Matrix Completion (L-BFGS, CPU)
├── scripts/
│   ├── run_phase2.py                  # Phase 2 orchestrator: probe -> tier -> anchor eval
│   ├── run_phase3.py                  # Phase 3 orchestrator: cascade + router training
│   ├── run_imc.py                     # IMC training (L-BFGS, <5 min CPU)
│   ├── validate_cascade.py            # DVR, GT agreement, cost R2, KS-test
│   ├── compute_bootstrap_cis.py       # Bootstrap 95% CIs for DVR and GT agreement
│   ├── oracle_gap_decomposition.py    # Decomposes oracle-router accuracy gap
│   ├── probe_sensitivity.py           # Probe model dropout and perturbation tests
│   ├── judge_sensitivity.py           # Neural judge flip-rate analysis
│   ├── run_imc_external_validation.py # IMC generalisation to new VLMs
│   ├── imc_dataset_holdout.py         # IMC cross-domain holdout experiment
│   ├── generate_routing_labels.py     # Derive routing labels from eval results
│   ├── oracle_arbitration.py          # Oracle arbiter pipeline
│   ├── cascade_mf_baseline.py         # Cascade-MF baseline (RouterBench comparison)
│   └── router/
│       ├── perceive_twophase.py       # Two-phase IPS router (train + inference)
│       ├── evaluate.py                # Router accuracy, cost, oracle-efficiency metrics
│       ├── baselines.py               # 13 baseline routing strategies
│       ├── data_loader.py             # Feature extraction from benchmark samples
│       └── config.py                  # Router hyperparameters and constants
├── tests/                     # Unit tests for scoring, dedup, anchor selector
├── data/croissant.json        # Croissant metadata (MLCommons format)
├── requirements.txt
├── LICENSE                    # CC BY 4.0
└── CITATION.bib
```

---

## Setup

Python 3.10+. No GPU required. All experiments run on CPU.

```bash
git clone https://github.com/bayesiansapien/perceive-benchmark
cd perceive-benchmark

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in the repo root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```

---

## Data download

The benchmark data is hosted on HuggingFace:

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='quantiphi-routing/perceive-benchmark',
    repo_type='dataset',
    local_dir='data/'
)
"
```

Source document images are **not redistributed** due to license constraints. Load them
directly from each source dataset's HuggingFace repository using the `image_path` field:

```bash
python src/ingestion/download_datasets.py   # downloads all 16 source datasets
```

---

## Reproducing paper claims

All numbered claims in the paper map to a specific script below. Pre-computed results
are already in `data/` so most scripts run without API calls.

| Claim | Value | Script | Needs API? |
|---|---|---|---|
| Cascade cost reduction | 60.7% | `scripts/validate_cascade.py` | No |
| Per-check DVR (n=171,392) | 8.2% | `scripts/validate_cascade.py` | No |
| DVR 95% CI (Wilson) | 8.0%–8.3% | `scripts/compute_bootstrap_cis.py` | No |
| GT label agreement (anchor, n=1,244) | 100% | `scripts/validate_cascade.py` | No |
| GT agreement 95% CI | 99.8%–100% | `scripts/compute_bootstrap_cis.py` | No |
| Verified-subset DVR (n=1,738) | 0.47% | `scripts/dvr_stratified.py` | No |
| Per-task DVR (T1–T6 breakdown) | 0.0–1.6% | `scripts/dvr_stratified.py` | No |
| Cost regret on verified subset | 99.6% zero | `scripts/dvr_stratified.py` | No |
| Stevens permutation null (canonical recovery) | 1/6 | `scripts/stevens_permutation_null.py` | No |
| IMC new-model AUC (Qwen3-VL-30B) | 0.833–0.845 | `scripts/run_imc_external_validation.py` | Yes |
| IMC new-model AUC (Llama-4-Scout) | 0.873 | `scripts/run_imc_external_validation.py` | Yes |
| IMC held-out queries AUC | 0.876 | `scripts/run_imc_external_validation.py` | No |
| IMC cross-domain AUC | 0.624 | `scripts/imc_dataset_holdout.py` | No |
| Router accuracy (PERCEIVE-IPS) | 61.6% | `scripts/train_perceive_router.py` | No |
| External baseline (k-NN router) | 51.1% | `scripts/knn_router_baseline.py` | No |
| External baseline (Cascade-MF) | 52.4% | `scripts/cascade_mf_baseline.py` | No |
| Oracle ceiling | 79.3% | `scripts/oracle_gap_decomposition.py` | No |
| Oracle gap: unroutable | 18.5% | `scripts/oracle_gap_decomposition.py` | No |
| Oracle gap: hard-routing | 13.0pp | `scripts/oracle_gap_decomposition.py` | No |
| Oracle gap: cost-tradeoff | 4.7pp | `scripts/oracle_gap_decomposition.py` | No |
| Probe tier stability (dropout) | 92.3–92.9% | `scripts/probe_sensitivity.py` | No |
| Probe tier stability (perturbation) | 90.3% ± 0.2% | `scripts/probe_sensitivity.py` | No |
| Judge routing-label flip rate | 5.6% | `scripts/judge_sensitivity.py` | No |

### Run all offline claims at once

```bash
bash scripts/run_full_pipeline.sh --eval-only
```

Skips all API phases. Reads pre-computed data from `data/`, runs oracle gap
decomposition, router evaluation, and all offline verification scripts, then
prints a formatted claim-verification table via `eval_paper_claims.py`.

To verify individual claims without running the full pipeline:

```bash
python scripts/eval_paper_claims.py
```

### Reproducing router accuracy (61.6%)

The PERCEIVE-IPS router uses text-only features (48 dimensions, no image encoder).
Train and evaluate in under 2 minutes on CPU:

```bash
python scripts/train_perceive_router.py          # 1 seed (~20 s)
python scripts/train_perceive_router.py --seeds 5  # 5-seed mean ± std (~90 s)
```

CLIP ViT-B/32 and MobileNetV3-Large embeddings (`data/embeddings/`) are included
for reproducibility research. Controlled ablation shows they are redundant with the
probe oracle at the 1,500-sample anchor scale and reduce accuracy by ~2pp when added
to the full text-feature set.

---

## Reproducing from scratch

> Total cost: ~USD 300 in API credits. Total wall time: ~48 hours on CPU.

**Phase 1: Data ingestion** (~2 hours, no API cost)

```bash
python src/ingestion/download_datasets.py
python scripts/run_phase2.py --water-test   # smoke test: 50 samples, ~$0.01
```

**Phase 2: Probe + anchor evaluation** (~36 hours, ~USD 200)

```bash
python scripts/run_phase2.py                # full run; checkpoint-aware, resumable
```

**Phase 3: Cascade, routing labels, router** (~12 hours, ~USD 100)

```bash
python scripts/run_phase3.py
python scripts/run_imc.py                   # IMC training: <5 min, no API
```

---

## Runtime and hardware

All experiments in the paper ran on a standard laptop CPU with no GPU.

| Component | Wall time | API cost |
|---|---|---|
| Phase 2 (probe + anchor) | ~36 hours | ~USD 200 |
| Phase 3 (cascade + router) | ~12 hours | ~USD 100 |
| IMC training (L-BFGS) | < 5 minutes | free |
| Router training | < 2 minutes | free |
| Evaluation scripts | < 10 minutes total | free |

---

## License

PERCEIVE annotations and code: **CC BY 4.0** (see `LICENSE`).

Source dataset images remain under their original licenses. Datasets with
non-commercial restrictions (RVL-CDIP: CC BY-SA 3.0 NC; VisualMRC: CC BY-NC-SA 4.0)
are referenced by path only. Images must be obtained from those datasets directly.

Per-dataset license details are in `data/croissant.json` (`dct:source` block).

---

## Citation

```bibtex
@inproceedings{perceive2026,
  title     = {{PERCEIVE}: A Self-Expanding Benchmark for Psychophysics-driven
               Elicitation of Routing Cost-Efficiency In Vision-Language Evaluation},
  author    = {Bhatti, Amit Singh and P M, Harikrishnan and Vaddina, Vishal},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://huggingface.co/datasets/quantiphi-routing/perceive-benchmark}
}
```
