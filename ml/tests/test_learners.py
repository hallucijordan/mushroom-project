"""Tests for all AbstractLearner implementations."""
import os
import tempfile
import numpy as np
import pytest
from mushroom_ml.learners import get_learner, LEARNER_REGISTRY

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_data():
    rng = np.random.default_rng(0)
    X = rng.random((80, 10)).astype(np.float32)
    y = rng.integers(0, 2, 80)
    return X, y


@pytest.mark.parametrize("key", list(LEARNER_REGISTRY.keys()))
class TestAllLearners:
    """Run the same suite against every registered learner."""

    def _make(self, key):
        # Use small params so tests run fast
        kwargs = {"n_estimators": 10} if key == "random_forest" else {}
        kwargs = {"n_estimators": 5} if key == "logistic" else kwargs
        return get_learner(key, **kwargs)

    def test_fit_returns_self(self, key, dummy_data):
        X, y = dummy_data
        learner = self._make(key)
        result = learner.fit(X, y)
        assert result is learner

    def test_predict_shape_and_values(self, key, dummy_data):
        X, y = dummy_data
        learner = self._make(key)
        learner.fit(X, y)
        pred = learner.predict(X)
        assert pred.shape == (len(X),)
        assert set(pred).issubset({0, 1})

    def test_predict_proba_shape(self, key, dummy_data):
        X, y = dummy_data
        learner = self._make(key)
        learner.fit(X, y)
        proba = learner.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_get_uncertainty_shape_and_nonneg(self, key, dummy_data):
        X, y = dummy_data
        learner = self._make(key)
        learner.fit(X, y)
        scores = learner.get_uncertainty(X)
        assert scores.shape == (len(X),)
        assert (scores >= 0).all()

    def test_predict_single_output_format(self, key, dummy_data):
        X, y = dummy_data
        learner = self._make(key)
        learner.fit(X, y)
        result = learner.predict_single(X[:1])
        assert result["label"] in {"edible", "poisonous"}
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["p_edible"] <= 1.0
        assert 0.0 <= result["p_poisonous"] <= 1.0
        assert abs(result["p_edible"] + result["p_poisonous"] - 1.0) < 1e-5

    def test_save_and_load(self, key, dummy_data):
        X, y = dummy_data
        learner = self._make(key)
        learner.fit(X, y)
        pred_before = learner.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            learner.save(path)
            loaded = type(learner).load(path)
            pred_after = loaded.predict(X)
            np.testing.assert_array_equal(pred_before, pred_after)
        finally:
            os.unlink(path)


# ── get_learner factory ────────────────────────────────────────────────────────

def test_get_learner_unknown_key():
    with pytest.raises(ValueError, match="Unknown learner"):
        get_learner("does_not_exist")


def test_learner_registry_keys():
    assert {"random_forest", "logistic", "mlp"} == set(LEARNER_REGISTRY.keys())
