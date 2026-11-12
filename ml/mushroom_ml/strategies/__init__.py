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