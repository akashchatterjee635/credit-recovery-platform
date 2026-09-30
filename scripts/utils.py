"""Reproducibility helpers shared by training and evaluation commands."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable
import uuid

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions(names: Iterable[str] | None = None) -> dict[str, str]:
    selected = names or [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "torch",
    ]
    versions = {}
    for name in selected:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def write_run_manifest(
    *,
    command: str,
    config: dict[str, Any],
    dataset_paths: Iterable[str | Path],
    output_paths: Iterable[str | Path],
    feature_order: list[str],
    split_ids: dict[str, list[int]] | None = None,
    metrics: dict[str, Any] | None = None,
    seed: int = 42,
    run_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    run_id = run_id or f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = Path("experiments") / "outputs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    datasets = {
        str(Path(path)): sha256_file(path)
        for path in dataset_paths
        if Path(path).exists()
    }
    outputs = {
        str(Path(path)): sha256_file(path)
        for path in output_paths
        if Path(path).exists()
    }
    manifest = {
        "run_id": run_id,
        "command": command,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "random_seeds": {"python": seed, "numpy": seed, "torch": seed},
        "dataset_hashes": datasets,
        "feature_list_and_order": feature_order,
        "split_identifiers": split_ids or {},
        "preprocessing_version": config.get("preprocessing_version", "unknown"),
        "package_versions": package_versions(),
        "hyperparameters": config,
        "evaluation_metrics": metrics or {},
        "artifact_checksums": outputs,
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return manifest, run_dir


def generate_manifest(script_name, outputs):
    """Compatibility wrapper used by older experiment scripts."""
    manifest, _ = write_run_manifest(
        command=script_name,
        config={},
        dataset_paths=[],
        output_paths=outputs,
        feature_order=[],
    )
    return manifest
