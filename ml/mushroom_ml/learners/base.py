from abc import ABC, abstractmethod
import numpy as np

class AbstractLearner(ABC):
    """Abstract model port """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AbstractLearner": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray: ...

    def predict_single(self, X: np.ndarray) -> dict:
        proba = self.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        return {
            "label": "edible" if label_idx == 0 else "poisonous",
            "confidence": float(proba[label_idx]),
            "p_edible": float(proba[0]),
            "p_poisonous": float(proba[1]),
        }

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "AbstractLearner": ...