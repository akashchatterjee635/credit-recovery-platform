import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import pytest
from experiments.synthetic_smoke_test.synthetic_builder import generate_mock_longitudinal_data
from backend.data.temporal_builder import assert_disjoint_applicant_splits
import inspect
import backend.data.temporal_builder as temporal_builder

def test_no_target_leakage_in_temporal_generation():
    # Provide a dataframe with TARGET=0 and TARGET=1
    df = pd.DataFrame({
        'SK_ID_CURR': [1, 2, 3, 4],
        'TARGET': [0, 1, 0, 1]
    })
    
    # We test that changing TARGET doesn't change the generated X_seq if seed is controlled.
    # Since generate_mock_longitudinal_data uses a fixed np.random.default_rng(42) internally
    # for vectorization/generation, running it twice on the SAME dataframe length 
    # but with different TARGETs should produce the EXACT SAME X_seq if there's no leakage!
    
    # NOTE: Since the current implementation instantiates rng inside the function, 
    # we just need to ensure the targets column isn't accessed. We can prove this by 
    # dropping the TARGET column. If it raises KeyError, it's leaked.
    
    df_no_target = df.drop(columns=['TARGET'])
    
    try:
        X_seq = generate_mock_longitudinal_data(df_no_target)
        assert X_seq.shape == (4, 36, 5)
    except KeyError as e:
        if 'TARGET' in str(e):
            pytest.fail("Leakage detected: temporal generation relies on TARGET column!")
        else:
            raise


def test_production_temporal_builder_does_not_reference_label():
    source = inspect.getsource(temporal_builder)
    forbidden = "TAR" + "GET"
    assert forbidden not in source


def test_applicant_splits_must_be_disjoint():
    with pytest.raises(ValueError, match="appears in both"):
        assert_disjoint_applicant_splits({"train": [1, 2], "test": [2, 3]})
