"""
Base learner for Judge A: a Random Forest with isotonic probability calibration.

train.py fits it on secondary_data.csv (61,069 simulated mushrooms) and saves it
to models/. ModelJudge loads it and classifies the features Gemini extracts from
a photo. Labels: 0 = edible, 1 = poisonous.

The first version of this file was never committed (the old .gitignore rule
"models/" also matched src/models/). This minimal version, written in 2026-11,
follows the design in milestone_1.md: RandomForest + CalibratedClassifierCV
(isotonic), tunable through the class constants below.
"""

import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.preprocessor import MODEL_DIR


class BaseLearner:
    N_ESTIMATORS = 300   # trees in the forest
    MAX_DEPTH = None     # grow each tree until its leaves are pure
    CV_FOLDS = 3         # folds used to fit the calibration
    RANDOM_STATE = 42
    FILENAME = "base_learner.joblib"

    def __init__(self):
        forest = RandomForestClassifier(
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            random_state=self.RANDOM_STATE,
            n_jobs=-1,
        )
        # A raw forest's vote share is not a reliable probability. Isotonic
        # calibration maps it to one, so the confidence shown in the UI means
        # something.
        self.model = CalibratedClassifierCV(forest, method="isotonic", cv=self.CV_FOLDS)

    def fit(self, X: np.ndarray, y) -> "BaseLearner":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(poisonous) for each row. Column 1 is class 1 = poisonous."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y) -> dict[str, float]:
        p_poisonous = self.predict_proba(X)
        pred = (p_poisonous >= 0.5).astype(int)
        return {
            "accuracy": accuracy_score(y, pred),
            "f1_poisonous": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, p_poisonous),
        }

    def predict_single(self, X: np.ndarray) -> dict:
        """Verdict for one row (the 1 x n array from feature_dict_to_array)."""
        p_poisonous = float(self.predict_proba(X)[0])
        p_edible = 1.0 - p_poisonous
        # A tie counts as poisonous: the safer mistake.
        label = "poisonous" if p_poisonous >= 0.5 else "edible"
        return {
            "label": label,
            "confidence": max(p_edible, p_poisonous),
            "p_edible": p_edible,
            "p_poisonous": p_poisonous,
        }

    # ── persistence ────────────────────────────────────────────────────────

    @classmethod
    def default_path(cls) -> str:
        return os.path.join(MODEL_DIR, cls.FILENAME)

    def save(self, path: str | None = None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self, path or self.default_path())

    @classmethod
    def load(cls, path: str | None = None) -> "BaseLearner":
        return joblib.load(path or cls.default_path())

    @classmethod
    def is_trained(cls) -> bool:
        return os.path.exists(cls.default_path())
