from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import backend.engine.explainer as explainer_module
from backend.engine.explainer import RiskExplainer
from backend.engine.feature_contract import FEATURE_CONTRACT_V3


def test_shap_unavailable_handled_gracefully(monkeypatch):
    monkeypatch.setattr(explainer_module, "SHAP_AVAILABLE", False)
    result = RiskExplainer(MagicMock(), {}).explain(pd.DataFrame())
    assert result["available"] is False
    assert "not installed" in result["message"]


def test_missing_raw_pipeline_is_explicit(monkeypatch):
    monkeypatch.setattr(explainer_module, "SHAP_AVAILABLE", True)
    adapter = MagicMock()
    adapter.get_preprocessor.return_value = None
    result = RiskExplainer(adapter, {}).explain(pd.DataFrame())
    assert result == {"available": False, "message": "Raw model pipeline is unavailable"}


def test_categorical_names_map_back_to_feature_contract(monkeypatch):
    monkeypatch.setattr(explainer_module, "SHAP_AVAILABLE", True)
    training = pd.DataFrame(
        {
            "AMT_CREDIT": [100_000.0, 200_000.0],
            "NAME_EDUCATION_TYPE": ["Secondary", "Higher education"],
        }
    )
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                ["AMT_CREDIT"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["NAME_EDUCATION_TYPE"],
            ),
        ]
    ).fit(training)
    adapter = MagicMock()
    adapter.get_preprocessor.return_value = preprocessor
    adapter._credit_transformer = None
    adapter._numeric_features = ["AMT_CREDIT"]
    adapter._cat_features = ["NAME_EDUCATION_TYPE"]
    shap_mock = MagicMock()
    shap_mock.shap_values.return_value = np.array([[0.1, -0.8, 0.2]])
    explainer = RiskExplainer(adapter, FEATURE_CONTRACT_V3)
    explainer._explainer = shap_mock
    result = explainer.explain(training.iloc[[0]], top_k=3)
    assert result["available"] is True
    categorical = [
        item for item in result["top_risk_drivers"] if item["feature"] == "NAME_EDUCATION_TYPE"
    ]
    assert categorical
    assert all("NAME_EDUCATION_TYPE" in item["transformed_feature"] for item in categorical)
    assert all(item["actionability"] == "IMMUTABLE" for item in categorical)
