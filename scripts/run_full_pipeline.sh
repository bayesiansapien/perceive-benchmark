#!/usr/bin/env bash
# PERCEIVE: full end-to-end pipeline
#
# Runs all four phases in sequence, then verifies all paper claims.
# Total wall time: ~48 hours on CPU. Total API cost: ~USD 300.
#
# Prerequisites:
#   - Python 3.10+ with requirements installed  (pip install -r requirements.txt)
#   - .env in repo root with API keys:
#       OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY
#   - Source datasets downloaded or downloadable from HuggingFace
#
# Usage:
#   bash scripts/run_full_pipeline.sh             # full run
#   bash scripts/run_full_pipeline.sh --eval-only # skip API phases, run verifications only
#   bash scripts/run_full_pipeline.sh --smoke-test # 50-sample smoke test (~$0.01)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Ensure scripts.* imports resolve regardless of how the script is invoked
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Prefer python3; fall back to python if python3 is not on PATH
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3 not found on PATH." >&2; exit 1
fi

# Load .env if present
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

EVAL_ONLY=0
SMOKE_TEST=0
for arg in "$@"; do
    case "$arg" in
        --eval-only)   EVAL_ONLY=1 ;;
        --smoke-test)  SMOKE_TEST=1 ;;
    esac
done

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }
hr()   { echo "$(printf '─%.0s' {1..72})"; }
skip() { log "SKIP: $1 (data not available; run the full pipeline first)"; }

# run_if_data FILE CMD...: runs CMD if FILE exists, else prints a skip message
run_if_data() {
    local guard="$1"; shift
    if [[ -f "$guard" ]]; then
        "$PYTHON" "$@"
    else
        skip "$guard missing"
    fi
}

hr
log "PERCEIVE benchmark pipeline"
log "Repo:   $REPO_ROOT"
log "Python: $PYTHON"
hr

# ── Phase 1: data ingestion ───────────────────────────────────────────────────
if [[ $EVAL_ONLY -eq 0 ]]; then
    log "Phase 1: downloading source datasets (~2 h, no API cost)"
    "$PYTHON" src/ingestion/download_datasets.py
    log "Phase 1 done."
    hr
fi

# ── Smoke test (optional) ─────────────────────────────────────────────────────
if [[ $SMOKE_TEST -eq 1 ]]; then
    log "Smoke test: 50 samples (~\$0.01)"
    "$PYTHON" scripts/run_phase2.py --water-test
    log "Smoke test done. Exiting."
    exit 0
fi

# ── Phase 2: probe + anchor evaluation ───────────────────────────────────────
if [[ $EVAL_ONLY -eq 0 ]]; then
    log "Phase 2: probe annotation + anchor evaluation (~36 h, ~USD 200)"
    log "Checkpoint-aware, safe to interrupt and resume."
    "$PYTHON" scripts/run_phase2.py
    log "Phase 2 done."
    hr
fi

# ── Phase 3: cascade + routing labels + router training ──────────────────────
if [[ $EVAL_ONLY -eq 0 ]]; then
    log "Phase 3: cascade evaluation, routing labels, router training (~12 h, ~USD 100)"
    "$PYTHON" scripts/run_phase3.py --stage 1a
    "$PYTHON" scripts/run_phase3.py --merge --split anchor
    "$PYTHON" scripts/run_phase3.py --cascade-validate
    "$PYTHON" scripts/generate_routing_labels.py
    log "Phase 3 done."
    hr
fi

# ── IMC training (CPU only, <5 min) ──────────────────────────────────────────
log "IMC: Inductive Matrix Completion training and validation (<5 min, no API)"
run_if_data "data/model_eval_results/final_eval_correct.jsonl" \
    scripts/run_imc.py
log "IMC done. Results in results/imc/imc_results.json"
hr

# ── Cascade validation ────────────────────────────────────────────────────────
log "Cascade validation: DVR, GT agreement, cost R2, KS test"
run_if_data "data/model_eval_results/merged/anchor_results.jsonl" \
    scripts/validate_cascade.py
hr

# ── Bootstrap confidence intervals ────────────────────────────────────────────
log "Bootstrap 95% CIs for DVR and GT agreement"
run_if_data "data/model_eval_results/merged/anchor_results.jsonl" \
    scripts/compute_bootstrap_cis.py
hr

# ── Oracle gap decomposition ──────────────────────────────────────────────────
log "Oracle gap decomposition"
run_if_data "data/routing_labels/routing_labels.jsonl" \
    scripts/oracle_gap_decomposition.py
hr

# ── Probe sensitivity ─────────────────────────────────────────────────────────
log "Probe sensitivity"
run_if_data "data/processed/probe_results.jsonl" \
    scripts/probe_sensitivity.py
hr

# ── Judge sensitivity (note: API keys needed for a live re-run) ───────────────
log "Judge sensitivity"
echo "  Pre-computed: data/judge_sensitivity_gpt54_report.json"
echo "  Pre-computed: data/judge_sensitivity_sonnet_report.json"
echo "  Live re-run:  python3 scripts/judge_sensitivity.py --model gpt54  (needs OPENAI_API_KEY)"
hr

# ── IMC external validation ────────────────────────────────────────────────────
log "IMC external validation (pre-computed)"
echo "  Results: data/imc_external_validation/imc_report.json"
echo "  Live re-run: python3 scripts/run_imc_external_validation.py  (needs API keys)"
hr

# ── IMC dataset holdout ───────────────────────────────────────────────────────
log "IMC dataset holdout (pre-computed)"
echo "  Results: data/imc_dataset_holdout_report.json"
hr

# ── Verify all paper claims ───────────────────────────────────────────────────
log "Verifying all paper claims against pre-computed artefacts"
"$PYTHON" scripts/eval_paper_claims.py
hr

log "Pipeline complete."
