import numpy as np
from mushroom_ml.learners.base import AbstractLearner
from mushroom_ml.strategies.base import AbstractStrategy

_N_MC = 10


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


class BALDSampling(AbstractStrategy):
    """Bayesian Active Learning by Disagreement (BALD).

    Score = H[y|x] - E_w[H[y|x,w]]
          = entropy of mean prediction - mean entropy of MC predictions
    """
    name = "BALD"
    key = "bald"

    def query(self, learner: AbstractLearner, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        # Collect MC samples of predict_proba
        # For learners that support it, call _mc_predict_proba; otherwise repeat predict_proba.
        if hasattr(learner, "_mc_predict_proba"):
            mc_probas = learner._mc_predict_proba(X_pool, n_passes=_N_MC)  # (MC, N, C)
        else:
            single = learner.predict_proba(X_pool)
            mc_probas = np.stack([single] * _N_MC)

        mean_proba = mc_probas.mean(axis=0)               # (N, C)
        h_mean = _entropy(mean_proba)                     # H[y|x]
        mean_h = _entropy(mc_probas).mean(axis=0)         # E[H[y|x,w]]
        bald_score = h_mean - mean_h

        return np.argsort(bald_score)[-n_samples:]
