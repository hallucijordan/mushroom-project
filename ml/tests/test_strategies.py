"""Tests for all AbstractStrategy implementations."""
import numpy as np
import pytest
from mushroom_ml.learners import get_learner
from mushroom_ml.strategies import get_strategy, STRATEGY_REGISTRY


@pytest.fixture
def trained_learner():
    rng = np.random.default_rng(1)
    X = rng.random((100, 10)).astype(np.float32)
    y = rng.integers(0, 2, 100)
    learner = get_learner("random_forest", n_estimators=10)
    learner.fit(X, y)
    learner._X_train = X  # needed by QueryByCommittee
    learner._y_train = y
    return learner, X


@pytest.mark.parametrize("key", list(STRATEGY_REGISTRY.keys()))
class TestAllStrategies:

    def test_query_returns_correct_count(self, key, trained_learner):
        learner, X_pool = trained_learner
        strategy = get_strategy(key)
        n = 10
        indices = strategy.query(learner, X_pool, n)
        assert len(indices) == n

    def test_query_indices_in_range(self, key, trained_learner):
        learner, X_pool = trained_learner
        strategy = get_strategy(key)
        indices = strategy.query(learner, X_pool, 10)
        assert ((indices >= 0) & (indices < len(X_pool))).all()

    def test_query_no_duplicates(self, key, trained_learner):
        learner, X_pool = trained_learner
        strategy = get_strategy(key)
        indices = strategy.query(learner, X_pool, 10)
        assert len(set(indices)) == len(indices)

    def test_query_returns_ndarray(self, key, trained_learner):
        learner, X_pool = trained_learner
        strategy = get_strategy(key)
        indices = strategy.query(learner, X_pool, 5)
        assert isinstance(indices, np.ndarray)


def test_get_strategy_unknown_key():
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("does_not_exist")


def test_strategy_registry_keys():
    assert {"random", "uncertainty", "committee", "bald"} == set(STRATEGY_REGISTRY.keys())
