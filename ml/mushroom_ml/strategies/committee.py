import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mushroom_ml.learners.base import AbstractLearner
from mushroom_ml.strategies.base import AbstractStrategy

_N_COMMITTEE = 5


class QueryByCommittee(AbstractStrategy):
    name = "Query by Committee"
    key = "committee"

    def query(self, learner: AbstractLearner, X_pool: np.ndarray, n_samples: int) -> np.ndarray:
        # Train 5 bootstrap RF models, measure vote disagreement
        n = len(X_pool)
        votes = np.zeros((n, _N_COMMITTEE), dtype=int)

        # We need labeled data to train the committee — learner must expose it,
        # or we fall back to the learner's own uncertainty.
        # If learner has _X_train/_y_train, use them; otherwise fall back.
        if hasattr(learner, "_X_train") and hasattr(learner, "_y_train"):
            X_tr, y_tr = learner._X_train, learner._y_train
            for i in range(_N_COMMITTEE):
                idx = np.random.choice(len(X_tr), len(X_tr), replace=True)
                m = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
                m.fit(X_tr[idx], y_tr[idx])
                votes[:, i] = m.predict(X_pool)
            # Disagreement = fraction of votes that differ from majority
            majority = np.apply_along_axis(
                lambda v: np.bincount(v).argmax(), axis=1, arr=votes
            )
            disagreement = (votes != majority[:, None]).mean(axis=1)
        else:
            disagreement = learner.get_uncertainty(X_pool)

        return np.argsort(disagreement)[-n_samples:]
