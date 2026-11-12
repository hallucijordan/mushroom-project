import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from mushroom_ml.learners.base import AbstractLearner


class MLPLearner(AbstractLearner):
    key = "mlp"
    name = "MLP"

    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=500, random_state=42):
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPLearner":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        # Entropy of predicted distribution
        proba = np.clip(self.predict_proba(X), 1e-12, 1.0)
        return -np.sum(proba * np.log(proba), axis=1)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MLPLearner":
        return joblib.load(path)
