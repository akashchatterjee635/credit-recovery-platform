"""Deterministic static training entry point, including a tiny CI smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import git_commit, set_random_seeds, sha256_file, write_run_manifest


def _load_config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def _smoke_frame(seed: int, rows: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.uniform(40_000, 250_000, rows)
    annuity = rng.uniform(4_000, 30_000, rows)
    debt = rng.uniform(0, 300_000, rows)
    late = rng.uniform(0, 0.7, rows)
    logits = -2.0 + 4.0 * late + 1.2 * debt / income + 0.8 * annuity / income
    probability = 1.0 / (1.0 + np.exp(-logits))
    return pd.DataFrame(
        {
            "AMT_CREDIT": rng.uniform(50_000, 600_000, rows),
            "AMT_INCOME_TOTAL": income,
            "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": -rng.integers(8_000, 25_000, rows),
            "DAYS_EMPLOYED": -rng.integers(30, 8_000, rows),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary", "Higher education", "Incomplete higher"], rows
            ),
            "BUREAU_TOTAL_DEBT": debt,
            "INST_LATE_RATIO": late,
            "TARGET": rng.binomial(1, probability),
        }
    )


def train(config: dict, smoke: bool = False) -> Path:
    seed = int(config.get("seed", 42))
    set_random_seeds(seed)
    dataset_path = Path(config.get("dataset", "data/train_reference.csv"))
    frame = _smoke_frame(seed) if smoke else pd.read_csv(dataset_path)
    label = config.get("label", "TARGET")
    features = list(config.get("features") or [column for column in frame if column != label])
    categorical = [column for column in features if not pd.api.types.is_numeric_dtype(frame[column])]
    numeric = [column for column in features if column not in categorical]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    try:
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            n_estimators=int(config.get("n_estimators", 30 if smoke else 300)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            random_state=seed,
            verbosity=-1,
        )
    except ImportError as exc:
        raise RuntimeError("Install the 'static' dependency group to train the model") from exc

    pipeline = Pipeline([("prep", preprocessor), ("clf", classifier)])
    pipeline.fit(frame[features], frame[label].astype(int))
    probability = pipeline.predict_proba(frame[features])[:, 1]
    metrics = {"train_roc_auc": float(roc_auc_score(frame[label], probability))}

    output = Path(
        config.get("smoke_output", "artifacts/demo_static_model.joblib")
        if smoke
        else config.get("output", "backend/models/lgbm_model.pkl")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "calibrated_model": pipeline,
            "raw_pipeline": pipeline,
            "credit_tf": None,
            "numeric_features": numeric,
            "cat_features": categorical,
        },
        output,
    )
    artifact_manifest = {
        "model_version": "lgbm-v3-calibrated" if not smoke else "lgbm-demo-smoke-v1",
        "git_commit": git_commit(),
        "dataset_hash": hashlib.sha256(
            frame.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "feature_contract_version": "feature-contract-v3",
        "constraint_registry_version": "constraint-registry-v2",
        "test_metrics": metrics,
        "artifact_sha256": sha256_file(output),
    }
    artifact_manifest_path = output.parent / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2), encoding="utf-8"
    )
    write_run_manifest(
        command="python -m scripts.train_static",
        config=config,
        dataset_paths=[] if smoke else [dataset_path],
        output_paths=[output, artifact_manifest_path],
        feature_order=features,
        metrics=metrics,
        seed=seed,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = train(_load_config(args.config), smoke=args.smoke)
    print(f"Saved deterministic static artifact to {output}")


if __name__ == "__main__":
    main()
