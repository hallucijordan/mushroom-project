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