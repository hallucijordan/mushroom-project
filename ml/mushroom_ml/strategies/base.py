# ml/mushroom_ml/strategies/base.py
from abc import ABC, abstractmethod
import numpy as np
from mushroom_ml.learners.base import AbstractLearner

class AbstractStrategy(ABC):
    """Interface for active learning query strategies."""

    name: str  # Display name, e.g., "Uncertainty Sampling"
    key: str   # Machine key, e.g., "uncertainty"

    @abstractmethod
    def query(
        self,
        learner: AbstractLearner,
        X_pool: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Select indices from X_pool to label next.

        Parameters
        ----------
        learner : trained model (has predict_proba, get_uncertainty)
        X_pool  : unlabeled pool feature matrix
        n_samples : how many to select

        Returns
        -------
        np.ndarray of integer indices into X_pool
        """
        ...