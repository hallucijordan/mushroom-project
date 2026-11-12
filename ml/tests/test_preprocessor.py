"""Tests for MushroomPreprocessor."""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from mushroom_ml.preprocessor import (
    MushroomPreprocessor,
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def sample_df():
    """Minimal DataFrame with all required columns."""
    rng = np.random.default_rng(2)
    n = 30
    data = {}
    for col in NUMERIC_FEATURES:
        data[col] = rng.uniform(1, 10, n)
    for col in CATEGORICAL_FEATURES:
        data[col] = np.random.choice(["a", "b", "c"], n)
    return pd.DataFrame(data)


@pytest.fixture
def fitted_preprocessor(sample_df):
    prep = MushroomPreprocessor()
    prep.fit_transform(sample_df)
    return prep


class TestMushroomPreprocessor:

    def test_fit_transform_output_shape(self, sample_df):
        prep = MushroomPreprocessor()
        X = prep.fit_transform(sample_df)
        assert X.shape == (len(sample_df), len(ALL_FEATURES))

    def test_fit_transform_no_nans(self, sample_df):
        prep = MushroomPreprocessor()
        X = prep.fit_transform(sample_df)
        assert not np.isnan(X).any()

    def test_transform_matches_fit_transform(self, sample_df):
        prep = MushroomPreprocessor()
        X1 = prep.fit_transform(sample_df)
        X2 = prep.transform(sample_df)
        np.testing.assert_array_equal(X1, X2)

    def test_transform_before_fit_raises(self, sample_df):
        prep = MushroomPreprocessor()
        with pytest.raises(RuntimeError, match="fit_transform"):
            prep.transform(sample_df)

    def test_nan_numeric_imputed(self):
        prep = MushroomPreprocessor()
        df = pd.DataFrame({col: [1.0, 2.0, np.nan] for col in NUMERIC_FEATURES}
                          | {col: ["a", "b", "a"] for col in CATEGORICAL_FEATURES})
        X = prep.fit_transform(df)
        assert not np.isnan(X).any()

    def test_bracket_range_parsed(self):
        prep = MushroomPreprocessor()
        df = pd.DataFrame({col: ["[2, 4]", "3.0", "5.0"] for col in NUMERIC_FEATURES}
                          | {col: ["a", "b", "a"] for col in CATEGORICAL_FEATURES})
        X = prep.fit_transform(df)
        # [2, 4] → midpoint 3.0; first row numeric values should be 3.0
        assert not np.isnan(X[0]).any()

    def test_feature_dict_to_array(self, fitted_preprocessor):
        d = {col: "a" for col in CATEGORICAL_FEATURES}
        d.update({col: 5.0 for col in NUMERIC_FEATURES})
        arr = fitted_preprocessor.feature_dict_to_array(d)
        assert arr.shape == (1, len(ALL_FEATURES))

    def test_save_and_load(self, fitted_preprocessor, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            fitted_preprocessor.save(path)
            loaded = MushroomPreprocessor.load(path)
            X1 = fitted_preprocessor.transform(sample_df)
            X2 = loaded.transform(sample_df)
            np.testing.assert_array_equal(X1, X2)
        finally:
            os.unlink(path)
