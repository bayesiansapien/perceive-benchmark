"""
DocRouteBench — Result Merger

Merges API track (this VM) and GPU track (DGX) result files
into a single deduplicated JSONL for a given split.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def merge_results(
    api_path: str,
    gpu_path: str,
    output_path: str,
) -> int:
    """
    Merge API + GPU results. Deduplicate on (sample_id, config_id).
    Later files take precedence on conflict (GPU results appended after API).

    Returns:
        Number of records in merged output.
    """
    seen: dict[tuple[str, str], dict] = {}

    for path in [api_path, gpu_path]:
        p = Path(path)
        if not p.exists():
            log.warning("Result file not found, skipping: %s", path)
            continue

        loaded = 0
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    key = (d.get("sample_id", ""), d.get("config_id", ""))
                    seen[key] = d  # later file overwrites (GPU > API for same key)
                    loaded += 1
                except json.JSONDecodeError:
                    pass
        log.info("  Loaded %d records from %s", loaded, path)

    records = list(seen.values())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

    log.info("Merged %d records → %s", len(records), output_path)
    return len(records)
