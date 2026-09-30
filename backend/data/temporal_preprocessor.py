"""Persisted preprocessing shared by temporal training and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

TEMPORAL_PREPROCESSOR_VERSION = "temporal-preprocessor-v1"


class TemporalPreprocessor:
    PAD_ID = 0
    UNK_ID = 1

    def __init__(
        self,
        *,
        max_sequence_length: int = 36,
        temporal_feature_names: list[str] | None = None,
        artifact_version: str = TEMPORAL_PREPROCESSOR_VERSION,
    ):
        self.artifact_version = artifact_version
        self.max_sequence_length = max_sequence_length
        self.temporal_feature_names = list(temporal_feature_names or [])
        self.cat_cols: list[str] = []
        self.cont_cols: list[str] = []
        self.cat_mappings: dict[str, dict[str, int]] = {}
        self.cont_medians: dict[str, float] = {}
        self.cont_centers: dict[str, float] = {}
        self.cont_scales: dict[str, float] = {}
        self.temporal_medians: np.ndarray | None = None
        self.temporal_scales: np.ndarray | None = None
        self.is_fitted = False

    @staticmethod
    def _category_key(value: Any) -> str:
        if pd.isna(value):
            return "<MISSING>"
        return f"{type(value).__name__}:{value}"

    def fit(
        self,
        train_df: pd.DataFrame,
        sequence_values: np.ndarray | None = None,
        sequence_mask: np.ndarray | None = None,
    ) -> "TemporalPreprocessor":
        excluded = {"SK_ID_CURR", "TARGET"}
        columns = [column for column in train_df.columns if column not in excluded]
        self.cat_cols = []
        self.cont_cols = []
        self.cat_mappings = {}

        for column in columns:
            dtype = train_df[column].dtype
            low_cardinality_integer = (
                pd.api.types.is_integer_dtype(dtype)
                and train_df[column].nunique(dropna=True) < 15
            )
            if not is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or low_cardinality_integer:
                self.cat_cols.append(column)
            else:
                self.cont_cols.append(column)

        for column in self.cat_cols:
            keys = sorted(
                {self._category_key(value) for value in train_df[column] if pd.notna(value)}
            )
            self.cat_mappings[column] = {
                key: index + 2 for index, key in enumerate(keys)
            }

        for column in self.cont_cols:
            numeric = pd.to_numeric(train_df[column], errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            q25 = float(numeric.quantile(0.25)) if numeric.notna().any() else median
            q75 = float(numeric.quantile(0.75)) if numeric.notna().any() else median
            scale = q75 - q25
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = float(numeric.std()) if numeric.notna().any() else 1.0
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = 1.0
            self.cont_medians[column] = median
            self.cont_centers[column] = median
            self.cont_scales[column] = scale

        if sequence_values is not None:
            values = np.asarray(sequence_values, dtype=np.float64)
            if values.ndim != 3:
                raise ValueError("sequence_values must have shape (rows, months, features)")
            if self.temporal_feature_names and values.shape[2] != len(self.temporal_feature_names):
                raise ValueError("Temporal feature order does not match the configured contract")
            mask = (
                np.asarray(sequence_mask, dtype=bool)
                if sequence_mask is not None
                else np.ones(values.shape[:2], dtype=bool)
            )
            if mask.shape != values.shape[:2]:
                raise ValueError("sequence_mask shape must match the first two sequence dimensions")
            flattened = values[mask]
            if flattened.size:
                self.temporal_medians = np.nanmedian(flattened, axis=0)
                q25 = np.nanpercentile(flattened, 25, axis=0)
                q75 = np.nanpercentile(flattened, 75, axis=0)
                scales = q75 - q25
                scales[~np.isfinite(scales) | (scales <= 1e-6)] = 1.0
                self.temporal_scales = scales
            else:
                self.temporal_medians = np.zeros(values.shape[2], dtype=np.float64)
                self.temporal_scales = np.ones(values.shape[2], dtype=np.float64)
        elif self.temporal_feature_names:
            width = len(self.temporal_feature_names)
            self.temporal_medians = np.zeros(width, dtype=np.float64)
            self.temporal_scales = np.ones(width, dtype=np.float64)

        self.is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("TemporalPreprocessor has not been fitted")

    def transform_static(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        categorical = np.zeros((len(frame), len(self.cat_cols)), dtype=np.int64)
        for index, column in enumerate(self.cat_cols):
            series = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            categorical[:, index] = [
                self.PAD_ID
                if pd.isna(value)
                else self.cat_mappings[column].get(self._category_key(value), self.UNK_ID)
                for value in series
            ]

        continuous = np.zeros((len(frame), len(self.cont_cols)), dtype=np.float32)
        for index, column in enumerate(self.cont_cols):
            series = frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
            numeric = pd.to_numeric(series, errors="coerce").fillna(self.cont_medians[column])
            continuous[:, index] = (
                (numeric.to_numpy(dtype=np.float64) - self.cont_centers[column])
                / self.cont_scales[column]
            ).astype(np.float32)
        return continuous, categorical

    def transform_sequence(
        self, sequence_values: np.ndarray, sequence_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        values = np.asarray(sequence_values, dtype=np.float64)
        mask = np.asarray(sequence_mask, dtype=bool)
        if values.ndim != 3 or mask.shape != values.shape[:2]:
            raise ValueError("Expected sequence shape (rows, months, features) and matching mask")
        if values.shape[1] != self.max_sequence_length:
            raise ValueError(
                f"Expected sequence length {self.max_sequence_length}, got {values.shape[1]}"
            )
        if self.temporal_medians is None or values.shape[2] != len(self.temporal_medians):
            raise ValueError("Temporal feature order or width does not match fitted metadata")
        filled = np.where(np.isnan(values), self.temporal_medians.reshape(1, 1, -1), values)
        transformed = (filled - self.temporal_medians.reshape(1, 1, -1)) / self.temporal_scales.reshape(1, 1, -1)
        transformed[~mask] = 0.0
        return transformed.astype(np.float32), mask

    def get_cardinalities(self) -> list[int]:
        self._check_fitted()
        return [max(mapping.values(), default=self.UNK_ID) + 1 for mapping in self.cat_mappings.values()]

    def metadata(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "continuous_features": self.cont_cols,
            "categorical_features": self.cat_cols,
            "categorical_cardinalities": self.get_cardinalities(),
            "temporal_features": self.temporal_feature_names,
            "max_sequence_length": self.max_sequence_length,
            "padding_id": self.PAD_ID,
            "unknown_id": self.UNK_ID,
        }

    def save(self, path: str | Path, metadata_path: str | Path | None = None) -> None:
        self._check_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        metadata_output = Path(metadata_path) if metadata_path else output.with_suffix(".json")
        metadata_output.write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_version: str | None = TEMPORAL_PREPROCESSOR_VERSION,
    ) -> "TemporalPreprocessor":
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("Artifact is not a TemporalPreprocessor")
        if expected_version and loaded.artifact_version != expected_version:
            raise ValueError(
                f"Temporal artifact version mismatch: expected {expected_version}, "
                f"got {loaded.artifact_version}"
            )
        loaded._check_fitted()
        return loaded
