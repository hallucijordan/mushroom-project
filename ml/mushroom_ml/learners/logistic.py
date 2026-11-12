import numpy as np
import joblib
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from mushroom_ml.learners.base import AbstractLearner


class BaggedLogisticLearner(AbstractLearner):
    key = "logistic"
    name = "Bagged Logistic Regression"

    def __init__(self, n_estimators=10, random_state=42):
        self._model = BaggingClassifier(
            estimator=LogisticRegression(max_iter=1000),
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedLogisticLearner":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        # Entropy of predicted distribution
        proba = self.predict_proba(X)
        proba = np.clip(proba, 1e-12, 1.0)
        return -np.sum(proba * np.log(proba), axis=1)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaggedLogisticLearner":
        return joblib.load(path)
