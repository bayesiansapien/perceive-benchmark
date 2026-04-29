#!/usr/bin/env python3
"""
DocRouteBench Phase 2 — Pipeline Orchestrator
==============================================
Runs all 8 Phase 2 components in order with checkpoint-aware resumption.

Usage:
    python scripts/run_phase2.py              # full run
    python scripts/run_phase2.py --water-test # smoke test (50 samples, ~$0.01)
    python scripts/run_phase2.py --from C5   # resume from specific component
    python scripts/run_phase2.py --status     # show done / pending status

Component order:
    C2  dedup       src/ingestion/dedup.py
    C3  prior       src/sampling/structural_prior.py
    C4  prefilter   src/sampling/prefilter.py
    C5  api_probe   src/sampling/api_probe.py       [most expensive — resume-critical]
    C6  difficulty  src/sampling/difficulty_estimator.py
    C7  stratify    src/sampling/stratified_sampler.py
    C8  anchor      src/anchor_set/anchor_selector.py

Checkpoint logic:
    Each component writes a <name>.done file to data/phase2_checkpoints/ on
    successful completion.  If the file exists the component is skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_phase2")

# ── Checkpoint directory ──────────────────────────────────────────────────────
CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "phase2_checkpoints"

# ── Probe results path (for cost reporting) ───────────────────────────────────
PROBE_RESULTS_PATH = _PROJECT_ROOT / "data" / "processed" / "probe_results.jsonl"

# ── Sample count paths (for final summary) ────────────────────────────────────
COUNT_PATHS = {
    "deduped":     _PROJECT_ROOT / "data" / "processed" / "all_samples_deduped.jsonl",
    "candidates":  _PROJECT_ROOT / "data" / "processed" / "candidates_40k.jsonl",
    "scored":      _PROJECT_ROOT / "data" / "processed" / "scored_samples.jsonl",
    "benchmark":   _PROJECT_ROOT / "data" / "benchmark" / "benchmark_5000.jsonl",
    "anchor":      _PROJECT_ROOT / "data" / "anchor_set" / "anchor_ids.json",
    "validation":  _PROJECT_ROOT / "data" / "validation_set" / "validation_ids.json",
}


# ── Component registry ────────────────────────────────────────────────────────

@dataclass
class Component:
    """Metadata for a single Phase 2 pipeline component."""
    id: str                       # e.g. "C2"
    name: str                     # e.g. "dedup"
    module_path: str              # e.g. "src.ingestion.dedup"
    entry_fn: str                 # public function to call, e.g. "run_deduplication"
    checkpoint_name: str          # stem of .done file, e.g. "dedup"
    description: str              # one-line description
    output_paths: List[str] = field(default_factory=list)
    # kwargs injected at runtime (populated from CLI flags)
    runtime_kwargs: Dict = field(default_factory=dict)


COMPONENTS: List[Component] = [
    Component(
        id="C2",
        name="dedup",
        module_path="src.ingestion.dedup",
        entry_fn="run_deduplication",
        checkpoint_name="dedup",
        description="Cross-dataset deduplication (image hash + text hash)",
        output_paths=["data/processed/all_samples_deduped.jsonl"],
    ),
    Component(
        id="C3",
        name="prior",
        module_path="src.sampling.structural_prior",
        entry_fn="run_structural_prior",
        checkpoint_name="structural_prior",
        description="Structural prior scoring (layout, complexity signals)",
        output_paths=["data/processed/prior_scored.jsonl"],
    ),
    Component(
        id="C4",
        name="prefilter",
        module_path="src.sampling.prefilter",
        entry_fn="run_prefilter",
        checkpoint_name="prefilter",
        description="Candidate pre-filter → 40K candidate pool",
        output_paths=["data/processed/candidates_40k.jsonl"],
    ),
    Component(
        id="C5",
        name="api_probe",
        module_path="src.sampling.api_probe",
        entry_fn="run_api_probe",
        checkpoint_name="api_probe",
        description="API probe: GPT-5.2 + Gemini Flash on candidates (~$23)",
        output_paths=["data/processed/probe_results.jsonl"],
    ),
    Component(
        id="C6",
        name="difficulty",
        module_path="src.sampling.difficulty_estimator",
        entry_fn="run_difficulty_estimator",
        checkpoint_name="difficulty_estimator",
        description="Difficulty estimation: VDS/RDS/SES from probe results",
        output_paths=["data/processed/scored_samples.jsonl"],
    ),
    Component(
        id="C7",
        name="stratify",
        module_path="src.sampling.stratified_sampler",
        entry_fn="run_stratified_sampling",
        checkpoint_name="stratified_sampler",
        description="Stratified sampling → 5,000-sample benchmark",
        output_paths=["data/benchmark/benchmark_5000.jsonl"],
    ),
    Component(
        id="C8",
        name="anchor",
        module_path="src.anchor_set.anchor_selector",
        entry_fn="run_anchor_selection",
        checkpoint_name="anchor_selector",
        description="Submodular anchor selection → 1K anchor + 500 val",
        output_paths=[
            "data/anchor_set/anchor_ids.json",
            "data/validation_set/validation_ids.json",
            "data/anchor_set/selection_report.json",
        ],
    ),
]

# Map component ID and name → Component for fast lookup
_BY_ID: Dict[str, Component] = {c.id: c for c in COMPONENTS}
_BY_NAME: Dict[str, Component] = {c.name: c for c in COMPONENTS}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _done_path(component: Component) -> Path:
    return CHECKPOINT_DIR / f"{component.checkpoint_name}.done"


def _is_done(component: Component) -> bool:
    return _done_path(component).exists()


def _mark_done(component: Component) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    _done_path(component).write_text("done\n")


# ── Dynamic import helper ─────────────────────────────────────────────────────

def _import_entry_fn(component: Component) -> Callable:
    """
    Dynamically import and return the entry function for a component.
    Raises ImportError with a clear message if the module is not found.
    """
    import importlib
    try:
        mod = importlib.import_module(component.module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot import module '{component.module_path}' for component {component.id} "
            f"({component.name}). Is the file implemented? Original error: {exc}"
        ) from exc

    fn = getattr(mod, component.entry_fn, None)
    if fn is None:
        raise ImportError(
            f"Module '{component.module_path}' has no function '{component.entry_fn}'. "
            f"Check that the entry point is exported."
        )
    return fn


# ── File-count helpers ────────────────────────────────────────────────────────

def _count_jsonl(path: Path) -> Optional[int]:
    """Return number of non-empty lines in a JSONL file, or None if missing."""
    if not path.exists():
        return None
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_json_list(path: Path) -> Optional[int]:
    """Return length of a JSON list file, or None if missing."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else None


def _probe_total_cost(results_path: Path) -> float:
    """Sum cost_usd from probe_results.jsonl. Returns 0.0 if file missing."""
    if not results_path.exists():
        return 0.0
    total = 0.0
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    total += rec.get("cost_usd", 0.0)
                except json.JSONDecodeError:
                    pass
    return total


# ── Status display ────────────────────────────────────────────────────────────

def _show_status() -> None:
    """Print a table of component statuses (DONE / PENDING)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    col_w = [4, 12, 50, 10]
    header = f"{'ID':<{col_w[0]}}  {'Name':<{col_w[1]}}  {'Description':<{col_w[2]}}  {'Status':<{col_w[3]}}"
    sep = "-" * len(header)

    print()
    print("DocRouteBench Phase 2 — Component Status")
    print(sep)
    print(header)
    print(sep)

    for c in COMPONENTS:
        status = "DONE" if _is_done(c) else "PENDING"
        print(
            f"{c.id:<{col_w[0]}}  {c.name:<{col_w[1]}}  {c.description:<{col_w[2]}}  {status:<{col_w[3]}}"
        )
        if _is_done(c):
            for rel_path in c.output_paths:
                abs_path = _PROJECT_ROOT / rel_path
                if abs_path.suffix == ".jsonl":
                    n = _count_jsonl(abs_path)
                elif abs_path.suffix == ".json":
                    n = _count_json_list(abs_path)
                else:
                    n = None
                count_str = f"{n:,}" if n is not None else "n/a"
                print(f"      {rel_path}  [{count_str} records]")

    print(sep)
    total_cost = _probe_total_cost(PROBE_RESULTS_PATH)
    print(f"Probe API cost so far: ${total_cost:.4f}")
    print()


# ── Sample count summary ──────────────────────────────────────────────────────

def _show_counts() -> None:
    """Print final sample counts across pipeline stages."""
    deduped = _count_jsonl(COUNT_PATHS["deduped"])
    candidates = _count_jsonl(COUNT_PATHS["candidates"])
    scored = _count_jsonl(COUNT_PATHS["scored"])
    benchmark = _count_jsonl(COUNT_PATHS["benchmark"])
    anchor = _count_json_list(COUNT_PATHS["anchor"])
    validation = _count_json_list(COUNT_PATHS["validation"])

    def _fmt(n: Optional[int]) -> str:
        return f"{n:,}" if n is not None else "—"

    print()
    print("Sample counts through pipeline:")
    print(f"  Deduped      : {_fmt(deduped)}")
    print(f"  Candidates   : {_fmt(candidates)}")
    print(f"  Scored       : {_fmt(scored)}")
    print(f"  Benchmark    : {_fmt(benchmark)}")
    print(f"  Anchor set   : {_fmt(anchor)}")
    print(f"  Validation   : {_fmt(validation)}")
    print()


# ── Component runner ──────────────────────────────────────────────────────────

def _run_component(
    component: Component,
    water_test: bool,
    timings: Dict[str, float],
) -> bool:
    """
    Run a single component.

    Returns True on success, False on failure.
    On success, marks the checkpoint .done file.
    Timing is recorded in *timings* dict keyed by component.id.
    """
    log.info("")
    log.info("=" * 60)
    log.info("%s [%s] — %s", component.id, component.name, component.description)
    log.info("=" * 60)

    t0 = time.monotonic()

    try:
        fn = _import_entry_fn(component)
    except ImportError as exc:
        log.error("Import failed for %s: %s", component.id, exc)
        timings[component.id] = time.monotonic() - t0
        return False

    # Build kwargs: inject water-test / checkpoint_dir arguments where applicable
    kwargs: Dict = dict(component.runtime_kwargs)
    kwargs["checkpoint_dir"] = str(CHECKPOINT_DIR)

    if water_test:
        # Components may accept different kwarg names for the sample limit.
        for wt_kwarg in ("max_samples", "n_samples", "n_probe_samples"):
            limit = 20 if component.id == "C5" else 50
            kwargs[wt_kwarg] = limit
        # C8: scale down anchor/validation sizes for water-test (39 samples total)
        if component.id == "C8":
            kwargs["anchor_size"] = 15
            kwargs["validation_size"] = 10

    # Call the entry function, stripping unknown kwargs gracefully
    try:
        import inspect
        sig = inspect.signature(fn)
        accepted = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

        result = fn(**filtered_kwargs)
        elapsed = time.monotonic() - t0
        timings[component.id] = elapsed
        log.info("%s [%s] completed in %.1f s.", component.id, component.name, elapsed)

        # Write checkpoint (some components write their own; this is the fallback)
        if not _is_done(component):
            _mark_done(component)

        return True

    except Exception as exc:
        elapsed = time.monotonic() - t0
        timings[component.id] = elapsed
        log.error(
            "%s [%s] FAILED after %.1f s: %s",
            component.id, component.name, elapsed, exc,
            exc_info=True,
        )
        return False


# ── Main orchestrator ─────────────────────────────────────────────────────────

def _resolve_from_component(from_id: str) -> int:
    """
    Return the index in COMPONENTS to start from.
    Accepts "C2"–"C8" or the short name (e.g. "dedup", "anchor").
    Raises SystemExit with a helpful message on unknown input.
    """
    key = from_id.upper()
    if key in _BY_ID:
        return COMPONENTS.index(_BY_ID[key])
    key_lower = from_id.lower()
    if key_lower in _BY_NAME:
        return COMPONENTS.index(_BY_NAME[key_lower])
    valid = ", ".join(f"{c.id}/{c.name}" for c in COMPONENTS)
    log.error("Unknown component '%s'. Valid values: %s", from_id, valid)
    sys.exit(1)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_phase2",
        description="DocRouteBench Phase 2 Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_phase2.py                # full run
  python scripts/run_phase2.py --water-test   # 50-sample smoke test (~$0.01)
  python scripts/run_phase2.py --from C5      # resume from api_probe
  python scripts/run_phase2.py --status       # show component status table
""",
    )
    parser.add_argument(
        "--water-test",
        action="store_true",
        help="Smoke test: run with max_samples=50, n_probe_samples=20.",
    )
    parser.add_argument(
        "--from",
        dest="from_component",
        metavar="CX",
        default=None,
        help="Resume from a specific component (e.g. C5 or api_probe). "
             "Components before this are skipped regardless of checkpoint state.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print component status table and exit (no components run).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoints and re-run all selected components.",
    )
    args = parser.parse_args(argv)

    # ── Status-only mode ──────────────────────────────────────────────────────
    if args.status:
        _show_status()
        return 0

    # ── Determine start index ─────────────────────────────────────────────────
    start_idx = 0
    if args.from_component:
        start_idx = _resolve_from_component(args.from_component)
        log.info("Resuming from component %s.", COMPONENTS[start_idx].id)

    mode_label = "WATER-TEST" if args.water_test else "FULL"
    log.info("")
    log.info("DocRouteBench Phase 2 — %s RUN", mode_label)
    if args.water_test:
        log.info("Water-test mode: max_samples=50, n_probe_samples=20, expected cost ~$0.01")
    log.info("Checkpoint directory: %s", CHECKPOINT_DIR)
    log.info("")

    # ── Ensure checkpoint dir exists ──────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run components ────────────────────────────────────────────────────────
    timings: Dict[str, float] = {}
    failed: List[str] = []

    pipeline_t0 = time.monotonic()

    for i, component in enumerate(COMPONENTS):
        if i < start_idx:
            # Skipped via --from flag
            log.info("%s [%s] — skipped (--from %s)", component.id, component.name, COMPONENTS[start_idx].id)
            continue

        if not args.force and _is_done(component):
            log.info("%s [%s] — DONE (checkpoint exists, skipping)", component.id, component.name)
            timings[component.id] = 0.0
            continue

        if args.force and _is_done(component):
            log.info("%s [%s] — checkpoint exists but --force set; re-running.", component.id, component.name)
            _done_path(component).unlink(missing_ok=True)

        success = _run_component(component, water_test=args.water_test, timings=timings)
        if not success:
            failed.append(component.id)
            log.error("")
            log.error("Component %s failed. Stopping pipeline.", component.id)
            log.error("Fix the issue and re-run with:  python scripts/run_phase2.py --from %s", component.id)
            break

    pipeline_elapsed = time.monotonic() - pipeline_t0

    # ── Final report ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("PHASE 2 RUN SUMMARY")
    log.info("=" * 60)
    log.info("Total wall time: %.1f min", pipeline_elapsed / 60)
    log.info("")

    log.info("Component timings:")
    for c in COMPONENTS:
        if c.id in timings:
            t = timings[c.id]
            status = "DONE" if _is_done(c) else ("FAILED" if c.id in failed else "SKIPPED")
            log.info("  %s %-12s  %6.1f s   [%s]", c.id, c.name, t, status)

    total_cost = _probe_total_cost(PROBE_RESULTS_PATH)
    log.info("")
    log.info("Total API probe cost: $%.4f", total_cost)

    _show_counts()

    if failed:
        log.error("Pipeline stopped at: %s", ", ".join(failed))
        return 1

    log.info("All components completed successfully.")
    return 0


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())
