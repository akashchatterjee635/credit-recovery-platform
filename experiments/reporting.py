"""Machine-readable benchmark run persistence."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
from typing import Any

import pandas as pd

from scripts.utils import git_commit, package_versions


def save_benchmark_run(
    run_id: str,
    *,
    config: dict[str, Any],
    applicant_results: pd.DataFrame,
    summary: dict[str, Any],
    report_markdown: str,
) -> Path:
    run_dir = Path("experiments") / "outputs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        applicant_results.to_parquet(run_dir / "applicant_results.parquet", index=False)
    except ImportError:
        applicant_results.to_json(
            run_dir / "applicant_results.jsonl", orient="records", lines=True
        )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "git_commit": git_commit(),
                "python": sys.version,
                "platform": platform.platform(),
                "packages": package_versions(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(report_markdown, encoding="utf-8")
    return run_dir
