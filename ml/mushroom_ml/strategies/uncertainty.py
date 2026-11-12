import numpy as np
from mushroom_ml.learners.base import AbstractLearner
from mushroom_ml.strategies.base import AbstractStrategy


class UncertaintySampling(AbstractStrategy):
    name = "Uncertainty Sampling"
    key = "uncertainty"

    def query(self, learner: AbstractLearner, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        scores = learner.get_uncertainty(X_pool)
        return np.argsort(scores)[-n_samples:]
