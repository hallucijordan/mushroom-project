import numpy as np
from mushroom_ml.learners.base import AbstractLearner
from mushroom_ml.strategies.base import AbstractStrategy
from mushroom_ml.preprocessor import MushroomPreprocessor
from mushroom_ml.evaluation import compute_metrics

class ALSession:
    """Manages one active learning session's state.

    This is a pure ML object — no DB, no HTTP, no cat.
    The backend wraps this and adds persistence + cat logic.
    """

    def __init__(
        self,
        learner: AbstractLearner,
        strategy: AbstractStrategy,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        X_initial: np.ndarray,
        y_initial: np.ndarray,
    ):
        self.learner = learner
        self.strategy = strategy

        # Labeled set (grows over time)
        self.X_labeled = X_initial.copy()
        self.y_labeled = y_initial.copy()

        # Unlabeled pool (shrinks over time)
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy()  # Hidden from the "user" but known to us

        # History
        self.history: list[dict] = []

        # Initial training
        self.learner.fit(self.X_labeled, self.y_labeled)
        self._record_snapshot()

    def predict_sample(self, X: np.ndarray) -> dict:
        """Predict a single sample (shape (1, n_features))."""
        return self.learner.predict_single(X)

    def label_sample(self, pool_index: int) -> dict:
        """Reveal label for a pool sample, add to labeled set.

        Returns
        -------
        {
            "true_label": int,
            "predicted_label": int,
            "predicted_conf": float,
            "was_correct": bool,
        }
        """
        X_sample = self.X_pool[pool_index:pool_index+1]
        y_true = self.y_pool[pool_index]
        prediction = self.learner.predict_single(X_sample)

        # Move from pool to labeled
        self.X_labeled = np.vstack([self.X_labeled, X_sample])
        self.y_labeled = np.append(self.y_labeled, y_true)
        self.X_pool = np.delete(self.X_pool, pool_index, axis=0)
        self.y_pool = np.delete(self.y_pool, pool_index)

        predicted_int = 0 if prediction["label"] == "edible" else 1
        return {
            "true_label": int(y_true),
            "predicted_label": predicted_int,
            "predicted_conf": prediction["confidence"],
            "was_correct": predicted_int == int(y_true),
        }

    def query_batch(self, n_samples: int = 50) -> np.ndarray:
        """Use the strategy to select the next batch from the pool.
        Returns indices into self.X_pool."""
        n = min(n_samples, len(self.X_pool))
        return self.strategy.query(self.learner, self.X_pool, n)

    def retrain(self) -> dict:
        """Retrain the model on the current labeled set.
        Returns updated metrics."""
        self.learner.fit(self.X_labeled, self.y_labeled)
        snapshot = self._record_snapshot()
        return snapshot

    def add_external_sample(self, X: np.ndarray, y: int) -> None:
        """Add a sample from outside the pool (e.g., photo upload)."""
        self.X_labeled = np.vstack([self.X_labeled, X])
        self.y_labeled = np.append(self.y_labeled, y)

    def get_learning_curve(self) -> list[dict]:
        """Return history of {labeled_count, accuracy, ...} snapshots."""
        return self.history

    @property
    def labeled_count(self) -> int:
        return len(self.y_labeled)

    @property
    def pool_remaining(self) -> int:
        return len(self.y_pool)

    def _record_snapshot(self) -> dict:
        metrics = compute_metrics(self.learner, self.X_labeled, self.y_labeled)
        snapshot = {"labeled_count": self.labeled_count, **metrics}
        self.history.append(snapshot)
        return snapshot