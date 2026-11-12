import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from mushroom_ml.learners.base import AbstractLearner


class RandomForestLearner(AbstractLearner):
    key = "random_forest"
    name = "Random Forest"

    def __init__(self, n_estimators=300, max_depth=None, random_state=42):
        self._model = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestLearner":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        # 1 - max probability: low confidence = high uncertainty
        proba = self.predict_proba(X)
        return 1.0 - proba.max(axis=1)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestLearner":
        return joblib.load(path)
