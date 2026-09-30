import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from backend.data.temporal_preprocessor import TemporalPreprocessor


@pytest.fixture()
def training_frame():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "income": [10.0, 20.0, 30.0, 40.0],
            "category": ["a", "b", "a", "c"],
            "TARGET": [0, 1, 0, 1],
        }
    )


def test_categories_are_persisted_and_unknown_maps_to_unk(training_frame):
    preprocessor = TemporalPreprocessor().fit(training_frame)
    _, categorical = preprocessor.transform_static(
        pd.DataFrame({"income": [10.0, 10.0], "category": ["a", "never-seen"]})
    )
    assert categorical[0, 0] == preprocessor.cat_mappings["category"]["str:a"]
    assert categorical[1, 0] == preprocessor.UNK_ID


def test_one_row_matches_batch_and_row_order_does_not_matter(training_frame):
    preprocessor = TemporalPreprocessor().fit(training_frame)
    batch_cont, batch_cat = preprocessor.transform_static(training_frame)
    one_cont, one_cat = preprocessor.transform_static(training_frame.iloc[[2]])
    np.testing.assert_allclose(one_cont[0], batch_cont[2])
    np.testing.assert_array_equal(one_cat[0], batch_cat[2])
    shuffled = training_frame.sample(frac=1.0, random_state=8)
    shuffled_cont, shuffled_cat = preprocessor.transform_static(shuffled)
    order = np.argsort(shuffled.index.to_numpy())
    np.testing.assert_allclose(shuffled_cont[order], batch_cont)
    np.testing.assert_array_equal(shuffled_cat[order], batch_cat)


def test_sequence_padding_is_zero_and_version_mismatch_fails(training_frame):
    values = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
    mask = np.array([[False, True, True]] * 4)
    preprocessor = TemporalPreprocessor(
        max_sequence_length=3, temporal_feature_names=["one", "two"]
    ).fit(training_frame, values, mask)
    transformed, transformed_mask = preprocessor.transform_sequence(values, mask)
    assert np.all(transformed[:, 0] == 0)
    np.testing.assert_array_equal(transformed_mask, mask)
    output_dir = Path("work")
    output_dir.mkdir(exist_ok=True)
    path = output_dir / "test_temporal_preprocessor.joblib"
    metadata_path = path.with_suffix(".json")
    try:
        preprocessor.save(path)
        with pytest.raises(ValueError, match="version mismatch"):
            TemporalPreprocessor.load(path, expected_version="future-version")
    finally:
        path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)


def test_attention_ignores_padded_months():
    torch = pytest.importorskip("torch")
    from backend.models.deep.fusion import TemporalAttention

    attention = TemporalAttention(hidden_dim=2)
    attention.eval()
    first = torch.tensor([[[100.0, -100.0], [1.0, 2.0], [2.0, 3.0]]])
    second = first.clone()
    second[:, 0] = torch.tensor([-999.0, 999.0])
    mask = torch.tensor([[False, True, True]])
    with torch.no_grad():
        context_a, _ = attention(first, mask)
        context_b, _ = attention(second, mask)
    torch.testing.assert_close(context_a, context_b)
