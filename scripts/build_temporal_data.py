"""Build genuine pre-cutoff monthly tensors and a persisted preprocessor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.data.temporal_builder import (
    APPLICANT_ID,
    TEMPORAL_FEATURES,
    assert_disjoint_applicant_splits,
    build_monthly_histories,
)
from backend.data.temporal_preprocessor import (
    TEMPORAL_PREPROCESSOR_VERSION,
    TemporalPreprocessor,
)
from scripts.utils import write_run_manifest


def _config(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def build(config: dict) -> None:
    data_dir = Path(config.get("data_dir", "data"))
    tensor_dir = Path(config.get("tensor_dir", "data/tensors"))
    artifact_dir = Path(config.get("artifact_dir", "artifacts"))
    tensor_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    split_names = tuple(config.get("splits", ["train", "cal", "test"]))
    references = {
        name: pd.read_csv(data_dir / f"{name}_reference.csv") for name in split_names
    }
    split_ids = {
        name: frame[APPLICANT_ID].astype(int).tolist() for name, frame in references.items()
    }
    assert_disjoint_applicant_splits(split_ids)

    built = {}
    for name, ids in split_ids.items():
        built[name] = build_monthly_histories(
            data_dir,
            ids,
            max_sequence_length=int(config.get("max_sequence_length", 36)),
        )

    preprocessor = TemporalPreprocessor(
        max_sequence_length=int(config.get("max_sequence_length", 36)),
        temporal_feature_names=TEMPORAL_FEATURES,
    ).fit(
        references["train"],
        built["train"].sequences,
        built["train"].padding_mask,
    )
    preprocessor.save(
        artifact_dir / "temporal_preprocessor.joblib",
        artifact_dir / "temporal_preprocessor.json",
    )

    outputs: list[Path] = []
    for name in split_names:
        sequences, masks = preprocessor.transform_sequence(
            built[name].sequences, built[name].padding_mask
        )
        continuous, categorical = preprocessor.transform_static(references[name])
        arrays = {
            "X_seq": sequences,
            "mask": masks,
            "X_cont": continuous,
            "X_cat": categorical,
            "ids": built[name].applicant_ids,
        }
        if "TARGET" in references[name]:
            arrays["y"] = references[name]["TARGET"].to_numpy(dtype=np.float32)
        for suffix, values in arrays.items():
            path = tensor_dir / f"{name}_{suffix}.npy"
            np.save(path, values)
            outputs.append(path)

    metadata = {
        "preprocessor_version": TEMPORAL_PREPROCESSOR_VERSION,
        "continuous_features": preprocessor.cont_cols,
        "categorical_features": preprocessor.cat_cols,
        "num_continuous": len(preprocessor.cont_cols),
        "num_categorical": len(preprocessor.cat_cols),
        "cat_cardinalities": preprocessor.get_cardinalities(),
        "temporal_feature_names": TEMPORAL_FEATURES,
        "temporal_features": len(TEMPORAL_FEATURES),
        "max_seq_len": preprocessor.max_sequence_length,
    }
    metadata_path = artifact_dir / "temporal_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs.extend([artifact_dir / "temporal_preprocessor.joblib", metadata_path])
    write_run_manifest(
        command="python -m scripts.build_temporal_data",
        config=config,
        dataset_paths=[data_dir / f"{name}_reference.csv" for name in split_names],
        output_paths=outputs,
        feature_order=preprocessor.cont_cols + preprocessor.cat_cols + TEMPORAL_FEATURES,
        split_ids=split_ids,
        seed=int(config.get("seed", 42)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    build(_config(args.config))


if __name__ == "__main__":
    main()
