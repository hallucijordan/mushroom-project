# ML Module Packaging Design

## Goal

All machine learning code lives in a single `ml/` package (`mushroom_ml`) that can be:
- Installed as a Python package (`pip install -e ml/`)
- Versioned independently from the web app
- Swapped with improved models without touching the backend
- Tested in isolation

The backend imports `mushroom_ml` and calls its public API. It never touches sklearn/xgboost directly.

---

## Package Structure

```
ml/
├── mushroom_ml/
│   ├── __init__.py              # Public API exports
│   ├── preprocessor.py          # MushroomPreprocessor (migrated from src/data/)
│   ├── learners/
│   │   ├── __init__.py          # LearnerRegistry
│   │   ├── base.py              # AbstractLearner interface
│   │   ├── random_forest.py     # RandomForestLearner
│   │   ├── logistic.py          # BaggedLogisticLearner
│   │   └── mlp.py               # MLPLearner
│   ├── strategies/
│   │   ├── __init__.py          # StrategyRegistry
│   │   ├── base.py              # AbstractStrategy interface
│   │   ├── random.py            # RandomSampling
│   │   ├── uncertainty.py       # UncertaintySampling
│   │   ├── committee.py         # QueryByCommittee
│   │   └── bald.py              # BALDSampling
│   ├── active_learning.py       # ALSession orchestrator
│   └── evaluation.py            # Metrics computation
├── setup.py
├── pyproject.toml
└── tests/
    ├── test_preprocessor.py
    ├── test_learners.py
    ├── test_strategies.py
    └── test_al_session.py
```

---

## Abstract Interfaces

### AbstractLearner

```python
# ml/mushroom_ml/learners/base.py
from abc import ABC, abstractmethod
import numpy as np

class AbstractLearner(ABC):
    """Interface that all mushroom classifiers must implement."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AbstractLearner":
        """Train on labeled data. Returns self."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0=edible, 1=poisonous)."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability array of shape (n_samples, 2)."""
        ...

    def predict_single(self, X: np.ndarray) -> dict:
        """Convenience: predict one sample, return structured dict."""
        proba = self.predict_proba(X)[0]
        label_idx = int(np.argmax(proba))
        return {
            "label": "edible" if label_idx == 0 else "poisonous",
            "confidence": float(proba[label_idx]),
            "p_edible": float(proba[0]),
            "p_poisonous": float(proba[1]),
        }

    @abstractmethod
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Return uncertainty scores for each sample (higher = more uncertain).
        Used by uncertainty-based query strategies."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "AbstractLearner":
        ...
```

### AbstractStrategy

```python
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
```

---

## Concrete Implementations

### Learners

| Class | Key | Backend | Uncertainty Method |
|---|---|---|---|
| `RandomForestLearner` | `random_forest` | `CalibratedClassifierCV(RandomForestClassifier)` | 1 - max(predict_proba) |
| `BaggedLogisticLearner` | `logistic` | `BaggingClassifier(LogisticRegression)` | Entropy of predict_proba |
| `MLPLearner` | `mlp` | `MLPClassifier` with MC Dropout via multiple forward passes | Variance across MC samples |

### Strategies

| Class | Key | Selection Logic |
|---|---|---|
| `RandomSampling` | `random` | `np.random.choice(len(X_pool), n_samples)` |
| `UncertaintySampling` | `uncertainty` | Top-n by `learner.get_uncertainty(X_pool)` |
| `QueryByCommittee` | `committee` | Train 5 bootstrap models, select samples with max vote disagreement |
| `BALDSampling` | `bald` | Maximize mutual information: `H[y|x] - E[H[y|x,w]]` via MC approximation |

---

## Registry Pattern

Learners and strategies self-register so the backend can discover them dynamically.

```python
# ml/mushroom_ml/learners/__init__.py
from mushroom_ml.learners.random_forest import RandomForestLearner
from mushroom_ml.learners.logistic import BaggedLogisticLearner
from mushroom_ml.learners.mlp import MLPLearner

LEARNER_REGISTRY: dict[str, type] = {
    "random_forest": RandomForestLearner,
    "logistic": BaggedLogisticLearner,
    "mlp": MLPLearner,
}

def get_learner(key: str, **kwargs):
    """Factory: get a learner by key."""
    if key not in LEARNER_REGISTRY:
        raise ValueError(f"Unknown learner: {key}. Available: {list(LEARNER_REGISTRY)}")
    return LEARNER_REGISTRY[key](**kwargs)
```

```python
# ml/mushroom_ml/strategies/__init__.py
from mushroom_ml.strategies.random import RandomSampling
from mushroom_ml.strategies.uncertainty import UncertaintySampling
from mushroom_ml.strategies.committee import QueryByCommittee
from mushroom_ml.strategies.bald import BALDSampling

STRATEGY_REGISTRY: dict[str, type] = {
    "random": RandomSampling,
    "uncertainty": UncertaintySampling,
    "committee": QueryByCommittee,
    "bald": BALDSampling,
}

def get_strategy(key: str, **kwargs):
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {key}. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[key](**kwargs)
```

---

## ALSession — The Orchestrator

```python
# ml/mushroom_ml/active_learning.py
import numpy as np
from mushroom_ml.learners.base import AbstractLearner
from mushroom_ml.strategies.base import AbstractStrategy
from mushroom_ml.preprocessor import MushroomPreprocessor
from mushroom_ml.evaluation import compute_metrics

class ALSession:
    """Manages one active learning session's state.

    This is a pure ML object — no DB, no HTTP, no cat.
    The backend wraps this and adds persistence + cat logic.
    """

    def __init__(
        self,
        learner: AbstractLearner,
        strategy: AbstractStrategy,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        X_initial: np.ndarray,
        y_initial: np.ndarray,
    ):
        self.learner = learner
        self.strategy = strategy

        # Labeled set (grows over time)
        self.X_labeled = X_initial.copy()
        self.y_labeled = y_initial.copy()

        # Unlabeled pool (shrinks over time)
        self.X_pool = X_pool.copy()
        self.y_pool = y_pool.copy()  # Hidden from the "user" but known to us

        # History
        self.history: list[dict] = []

        # Initial training
        self.learner.fit(self.X_labeled, self.y_labeled)
        self._record_snapshot()

    def predict_sample(self, X: np.ndarray) -> dict:
        """Predict a single sample (shape (1, n_features))."""
        return self.learner.predict_single(X)

    def label_sample(self, pool_index: int) -> dict:
        """Reveal label for a pool sample, add to labeled set.

        Returns
        -------
        {
            "true_label": int,
            "predicted_label": int,
            "predicted_conf": float,
            "was_correct": bool,
        }
        """
        X_sample = self.X_pool[pool_index:pool_index+1]
        y_true = self.y_pool[pool_index]
        prediction = self.learner.predict_single(X_sample)

        # Move from pool to labeled
        self.X_labeled = np.vstack([self.X_labeled, X_sample])
        self.y_labeled = np.append(self.y_labeled, y_true)
        self.X_pool = np.delete(self.X_pool, pool_index, axis=0)
        self.y_pool = np.delete(self.y_pool, pool_index)

        predicted_int = 0 if prediction["label"] == "edible" else 1
        return {
            "true_label": int(y_true),
            "predicted_label": predicted_int,
            "predicted_conf": prediction["confidence"],
            "was_correct": predicted_int == int(y_true),
        }

    def query_batch(self, n_samples: int = 50) -> np.ndarray:
        """Use the strategy to select the next batch from the pool.
        Returns indices into self.X_pool."""
        n = min(n_samples, len(self.X_pool))
        return self.strategy.query(self.learner, self.X_pool, n)

    def retrain(self) -> dict:
        """Retrain the model on the current labeled set.
        Returns updated metrics."""
        self.learner.fit(self.X_labeled, self.y_labeled)
        snapshot = self._record_snapshot()
        return snapshot

    def add_external_sample(self, X: np.ndarray, y: int) -> None:
        """Add a sample from outside the pool (e.g., photo upload)."""
        self.X_labeled = np.vstack([self.X_labeled, X])
        self.y_labeled = np.append(self.y_labeled, y)

    def get_learning_curve(self) -> list[dict]:
        """Return history of {labeled_count, accuracy, ...} snapshots."""
        return self.history

    @property
    def labeled_count(self) -> int:
        return len(self.y_labeled)

    @property
    def pool_remaining(self) -> int:
        return len(self.y_pool)

    def _record_snapshot(self) -> dict:
        metrics = compute_metrics(self.learner, self.X_labeled, self.y_labeled)
        snapshot = {"labeled_count": self.labeled_count, **metrics}
        self.history.append(snapshot)
        return snapshot
```

---

## Packaging & Versioning

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mushroom-ml"
version = "0.1.0"
description = "ML package for mushroom edibility classification with active learning"
requires-python = ">=3.11"
dependencies = [
    "scikit-learn>=1.3",
    "xgboost>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]
```

### Installation

```bash
# Development (editable install)
pip install -e ml/

# Production (in Docker)
pip install ml/
```

### How the Backend Uses It

```python
# backend/app/services/al_engine.py
from mushroom_ml import ALSession, get_learner, get_strategy, MushroomPreprocessor

class ALEngine:
    def create_session(self, learner_key: str, strategy_key: str, init_method: str) -> str:
        preprocessor = MushroomPreprocessor.load()
        X_pool, y_pool = preprocessor.load_pool_data()

        learner = get_learner(learner_key)
        strategy = get_strategy(strategy_key)

        if init_method == "d_optimal":
            initial_indices = d_optimal_init(X_pool, n=200)
        else:
            initial_indices = random_init(X_pool, n=200)

        X_init = X_pool[initial_indices]
        y_init = y_pool[initial_indices]
        X_pool_remaining = np.delete(X_pool, initial_indices, axis=0)
        y_pool_remaining = np.delete(y_pool, initial_indices)

        session = ALSession(
            learner=learner,
            strategy=strategy,
            X_pool=X_pool_remaining,
            y_pool=y_pool_remaining,
            X_initial=X_init,
            y_initial=y_init,
        )

        session_id = self._persist(session)
        return session_id
```

---

## Migration Plan from Current Code

| Current File | Migrates To | Changes |
|---|---|---|
| `src/models/base_learner.py` | `ml/mushroom_ml/learners/random_forest.py` | Implement `AbstractLearner` interface, add `get_uncertainty()` |
| `src/data/preprocessor.py` | `ml/mushroom_ml/preprocessor.py` | Unchanged logic, new package path |
| `main.ipynb` (AL loop) | `ml/mushroom_ml/active_learning.py` | Refactored into `ALSession` class |
| `src/agents/gemini_extractor.py` | `backend/app/services/gemini_bridge.py` | Stays in backend (not ML, it's an API integration) |
| `src/judges/model_judge.py` | `backend/app/services/` | Rewritten to use `mushroom_ml` learner interface |
| `src/judges/direct_judge.py` | `backend/app/services/gemini_bridge.py` | Merged into Gemini bridge service |

---

## Adding a New Model Later

To add, say, an XGBoost learner:

1. Create `ml/mushroom_ml/learners/xgboost_learner.py`
2. Implement `AbstractLearner` (fit, predict, predict_proba, get_uncertainty, save, load)
3. Register in `ml/mushroom_ml/learners/__init__.py`
4. Bump version in `pyproject.toml`
5. The backend auto-discovers it via the registry — no backend code changes needed

Same pattern for a new query strategy.
