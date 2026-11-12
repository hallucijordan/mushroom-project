import os
import uuid
import numpy as np
from mushroom_ml import ALSession, get_learner, get_strategy, MushroomPreprocessor
from mushroom_ml.preprocessor import load_secondary
from backend.app.config import Settings

class ALEngine:
    """Manages all active training sessions in memory."""

    def __init__(self):
        self.sessions: dict[str, ALSession] = {}
        self._preprocessor = None
        self._X_full = None
        self._y_full = None
        self._settings = Settings()

    def _load_data(self):
        """Load and preprocess the full secondary dataset (once per process).

        If a saved preprocessor exists in model_dir, load it and transform.
        Otherwise fit a new one on the data and save it for future runs.
        """
        if self._preprocessor is not None:
            return

        X_raw, y = load_secondary(data_dir=self._settings.data_dir)

        preprocessor_path = os.path.join(self._settings.model_dir, "preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            self._preprocessor = MushroomPreprocessor.load(preprocessor_path)
            self._X_full = self._preprocessor.transform(X_raw)
        else:
            self._preprocessor = MushroomPreprocessor()
            self._X_full = self._preprocessor.fit_transform(X_raw)
            os.makedirs(self._settings.model_dir, exist_ok=True)
            self._preprocessor.save(preprocessor_path)

        self._y_full = y.values

    def create_session(self, learner_key: str, strategy_key: str, init_method: str) -> str:
        self._load_data()
        session_id = str(uuid.uuid4())

        learner = get_learner(learner_key)
        strategy = get_strategy(strategy_key)

        # 初始化标注集（200个样本）
        n_init = 200
        if init_method == "d_optimal":
            indices = self._d_optimal_init(self._X_full, n_init)
        else:
            indices = np.random.choice(len(self._X_full), n_init, replace=False)

        X_init = self._X_full[indices]
        y_init = self._y_full[indices]
        mask = np.ones(len(self._X_full), dtype=bool)
        mask[indices] = False
        X_pool = self._X_full[mask]
        y_pool = self._y_full[mask]

        self.sessions[session_id] = ALSession(
            learner=learner, strategy=strategy,
            X_pool=X_pool, y_pool=y_pool,
            X_initial=X_init, y_initial=y_init,
        )
        return session_id

    def predict_next(self, session_id: str) -> dict:
        session = self.sessions[session_id]
        # 用策略选1个样本
        indices = session.query_batch(n_samples=1)
        idx = indices[0]
        X_sample = session.X_pool[idx:idx+1]
        prediction = session.predict_sample(X_sample)
        return {"pool_index": int(idx), **prediction}

    def reveal_and_retrain(self, session_id: str, pool_index: int) -> dict:
        session = self.sessions[session_id]
        result = session.label_sample(pool_index)

        # Retrain every 50 newly labeled samples
        if session.labeled_count % 50 == 0:
            metrics = session.retrain()
            result["retrained"] = True
            result["metrics"] = metrics
        else:
            result["retrained"] = False

        return result

    def train_step(self, session_id: str) -> dict:
        """
        One atomic auto-training step:
          1. Query next sample from pool (using the session's strategy)
          2. Model predicts on that sample
          3. True label is revealed automatically (it's known in the dataset)
          4. Sample moves to labeled set; retrain every 50 labels
        Returns everything the frontend needs to animate one training round.
        """
        session = self.sessions[session_id]

        # 1. Pick next sample index using the active strategy
        indices = session.query_batch(n_samples=1)
        idx = int(indices[0])

        X_sample = session.X_pool[idx: idx + 1]
        true_label_int = int(session.y_pool[idx])
        true_label_str = "edible" if true_label_int == 0 else "poisonous"

        # 2. Model predicts
        prediction = session.predict_sample(X_sample)
        predicted_label_str = prediction["label"]
        was_correct = predicted_label_str == true_label_str

        # 3. Auto-reveal: move sample to labeled set, retrain if threshold hit
        session.label_sample(idx)
        retrained = False
        new_accuracy = session.get_learning_curve()[-1]["cv_accuracy"]

        if session.labeled_count % 50 == 0:
            metrics = session.retrain()
            retrained = True
            new_accuracy = metrics.get("cv_accuracy", new_accuracy)

        # 4. Choose cat reaction
        if was_correct:
            cat_reaction = "happy"
        elif true_label_str == "poisonous":
            cat_reaction = "sick"       # false negative — ate poison
        else:
            cat_reaction = "confused"   # false positive — needlessly avoided safe one

        # Build a minimal feature summary for the specimen card
        mushroom_features = self._sample_features(idx, session)

        return {
            "mushroom": {
                "index": idx,
                "features": mushroom_features,
            },
            "prediction": {
                "label":       predicted_label_str,
                "confidence":  prediction["confidence"],
                "p_edible":    prediction["p_edible"],
                "p_poisonous": prediction["p_poisonous"],
                "dialogue":    self._prediction_dialogue(prediction, mushroom_features),
            },
            "true_label":     true_label_str,
            "was_correct":    was_correct,
            "new_accuracy":   new_accuracy,
            "cat_reaction":   cat_reaction,
            "labeled_count":  session.labeled_count,
            "pool_remaining": session.pool_remaining,
            "retrained":      retrained,
        }

    def set_strategy(self, session_id: str, strategy_key: str) -> None:
        """Replace the active query strategy for a running session."""
        session = self.sessions[session_id]
        session.strategy = get_strategy(strategy_key)

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sample_features(pool_idx: int, session) -> dict:
        """Return a human-readable feature dict for a pool sample.

        The pool is a numpy array of encoded features; we return the
        raw index and a placeholder dict — the frontend already has
        column names and can display them from the specimen card.
        The real feature names come from the preprocessor's column list.
        """
        # If the preprocessor exposes column names, use them; otherwise
        # return an empty dict and the frontend will show the index only.
        try:
            cols = session.feature_names  # set by ALSession if available
            row = session.X_pool[pool_idx]
            return {c: str(round(float(v), 3)) for c, v in zip(cols, row)}
        except AttributeError:
            return {}

    @staticmethod
    def _prediction_dialogue(prediction: dict, features: dict) -> str:
        conf = prediction["confidence"]
        label = prediction["label"]
        feature_hint = next(iter(features.values()), "its appearance") if features else "its appearance"
        if conf >= 0.80:
            return f"Definitely {label}! {feature_hint} gives it away."
        elif conf >= 0.60:
            return f"I'll say {label}. {feature_hint} hints at it. ({round(conf * 100)}%)"
        else:
            return f"Hmm... I'll guess {label}? I'm not very sure. ({round(conf * 100)}%)"

    def get_state(self, session_id: str) -> dict:
        """Return current session stats (labeled count, pool size, latest metrics)."""
        session = self.sessions[session_id]
        latest = session.get_learning_curve()[-1]
        return {
            "labeled_count": session.labeled_count,
            "pool_remaining": session.pool_remaining,
            "cv_accuracy": latest["cv_accuracy"],
            "cv_std": latest["cv_std"],
        }