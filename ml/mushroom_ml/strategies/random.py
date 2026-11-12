import numpy as np
from mushroom_ml.learners.base import AbstractLearner
from mushroom_ml.strategies.base import AbstractStrategy


class RandomSampling(AbstractStrategy):
    name = "Random Sampling"
    key = "random"

    def query(self, learner: AbstractLearner, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        return np.random.choice(len(X_pool), size=n_samples, replace=False)
