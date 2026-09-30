"""SHAP explanations mapped back to the public feature contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    shap = None
    SHAP_AVAILABLE = False


class RiskExplainer:
    def __init__(self, risk_adapter, feature_contract):
        self.adapter = risk_adapter
        self.feature_contract = feature_contract
        self._explainer = None

    def _init_explainer(self) -> None:
        if not SHAP_AVAILABLE:
            return
        raw_model = self.adapter.get_raw_lgbm()
        if raw_model is None:
            return
        try:
            self._explainer = shap.TreeExplainer(raw_model)
        except Exception:
            self._explainer = None

    @staticmethod
    def _pipeline_output_names(transformer, columns: list[str]) -> list[str]:
        if hasattr(transformer, "get_feature_names_out"):
            try:
                return [str(name) for name in transformer.get_feature_names_out(columns)]
            except Exception:
                pass
        named_steps = getattr(transformer, "named_steps", {})
        final = next(reversed(named_steps.values()), None) if named_steps else None
        if final is not None and hasattr(final, "get_feature_names_out"):
            try:
                return [str(name) for name in final.get_feature_names_out(columns)]
            except Exception:
                pass
        return list(columns)

    @classmethod
    def _feature_names(cls, preprocessor) -> tuple[list[str], list[str]]:
        transformed: list[str] = []
        origins: list[str] = []
        for name, transformer, columns in getattr(preprocessor, "transformers_", []):
            if name == "remainder" and transformer == "drop":
                continue
            cols = [str(column) for column in columns]
            outputs = cls._pipeline_output_names(transformer, cols)
            for output in outputs:
                cleaned = output.split("__", 1)[-1]
                origin = next(
                    (column for column in sorted(cols, key=len, reverse=True)
                     if cleaned == column or cleaned.startswith(f"{column}_")),
                    cleaned,
                )
                transformed.append(cleaned)
                origins.append(origin)
        return transformed, origins

    @staticmethod
    def _first_row_shap_values(values) -> np.ndarray:
        if isinstance(values, list):
            values = values[-1]
        array = np.asarray(values)
        if array.ndim == 3:
            array = array[:, :, -1]
        if array.ndim == 2:
            array = array[0]
        return np.asarray(array, dtype=float).reshape(-1)

    def explain(self, applicant_df: pd.DataFrame, top_k: int = 5) -> dict:
        if not SHAP_AVAILABLE:
            return {"available": False, "message": "shap not installed"}

        preprocessor = self.adapter.get_preprocessor()
        if preprocessor is None:
            return {"available": False, "message": "Raw model pipeline is unavailable"}
        if self._explainer is None:
            self._init_explainer()
        if self._explainer is None:
            return {"available": False, "message": "Could not initialize SHAP explainer"}

        try:
            frame = applicant_df.copy()
            credit_transformer = getattr(self.adapter, "_credit_transformer", None)
            if credit_transformer:
                frame = credit_transformer.transform(frame)
            all_columns = list(getattr(self.adapter, "_numeric_features", [])) + list(
                getattr(self.adapter, "_cat_features", [])
            )
            for column in all_columns:
                if column not in frame.columns:
                    frame[column] = np.nan
            frame = frame[all_columns]

            matrix = preprocessor.transform(frame)
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            transformed_names, origins = self._feature_names(preprocessor)
            width = matrix.shape[1]
            if len(transformed_names) != width:
                transformed_names = [f"feature_{index}" for index in range(width)]
                origins = list(transformed_names)
            transformed_frame = pd.DataFrame(matrix, columns=transformed_names)
            values = self._first_row_shap_values(
                self._explainer.shap_values(transformed_frame)
            )

            drivers = []
            for index in np.argsort(np.abs(values))[::-1][:top_k]:
                transformed_name = transformed_names[index]
                origin = origins[index]
                contract_entry = self.feature_contract.get(origin)
                drivers.append(
                    {
                        "feature": origin,
                        "transformed_feature": transformed_name,
                        "contribution": round(float(values[index]), 4),
                        "direction": (
                            "increases_risk" if values[index] > 0 else "decreases_risk"
                        ),
                        "actionability": (
                            contract_entry.feature_class if contract_entry else "UNKNOWN"
                        ),
                        "action": (
                            contract_entry.corresponding_action if contract_entry else ""
                        ),
                    }
                )
            return {"available": True, "top_risk_drivers": drivers}
        except Exception as exc:
            return {"available": False, "message": f"SHAP error: {exc}"}
