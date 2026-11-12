"""Tests for ALSession (active learning loop)."""
import numpy as np
import pytest
from mushroom_ml.active_learning import ALSession
from mushroom_ml.learners import get_learner
from mushroom_ml.strategies import get_strategy


@pytest.fixture
def session():
    rng = np.random.default_rng(3)
    X = rng.random((200, 10)).astype(np.float32)
    y = rng.integers(0, 2, 200)
    X_init, y_init = X[:20], y[:20]
    X_pool, y_pool = X[20:], y[20:]
    learner = get_learner("random_forest", n_estimators=10)
    strategy = get_strategy("random")
    return ALSession(learner, strategy, X_pool, y_pool, X_init, y_init)


class TestALSession:

    def test_initial_labeled_count(self, session):
        assert session.labeled_count == 20

    def test_initial_pool_size(self, session):
        assert session.pool_remaining == 180

    def test_history_recorded_after_init(self, session):
        assert len(session.history) == 1
        assert "cv_accuracy" in session.history[0]

    def test_query_batch_returns_valid_indices(self, session):
        indices = session.query_batch(n_samples=10)
        assert len(indices) == 10
        assert ((indices >= 0) & (indices < session.pool_remaining)).all()

    def test_label_sample_moves_from_pool(self, session):
        pool_before = session.pool_remaining
        labeled_before = session.labeled_count
        result = session.label_sample(0)
        assert session.pool_remaining == pool_before - 1
        assert session.labeled_count == labeled_before + 1

    def test_label_sample_result_format(self, session):
        result = session.label_sample(0)
        assert result["true_label"] in {0, 1}
        assert result["predicted_label"] in {0, 1}
        assert 0.0 <= result["predicted_conf"] <= 1.0
        assert isinstance(result["was_correct"], bool)

    def test_retrain_appends_history(self, session):
        session.label_sample(0)
        session.retrain()
        assert len(session.history) == 2

    def test_predict_sample_output_format(self, session):
        X_sample = session.X_pool[:1]
        result = session.predict_sample(X_sample)
        assert result["label"] in {"edible", "poisonous"}
        assert 0.0 <= result["confidence"] <= 1.0

    def test_add_external_sample_grows_labeled(self, session):
        rng = np.random.default_rng(99)
        X_new = rng.random((1, 10)).astype(np.float32)
        before = session.labeled_count
        session.add_external_sample(X_new, 1)
        assert session.labeled_count == before + 1

    def test_get_learning_curve_structure(self, session):
        curve = session.get_learning_curve()
        assert isinstance(curve, list)
        assert "labeled_count" in curve[0]
        assert "cv_accuracy" in curve[0]
