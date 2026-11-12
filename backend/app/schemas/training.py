"""Pydantic schemas for the /api/training endpoints."""
from pydantic import BaseModel
from typing import Optional


# ── POST /training/start ──────────────────────────────────────────────────────

class StartRequest(BaseModel):
    learner_type: str = "random_forest"   # "random_forest" | "logistic" | "mlp"
    strategy: str = "random"             # "random" | "uncertainty" | "committee" | "bald"
    init_strategy: str = "random"        # "random" | "d_optimal"
    cat_name: str = "Professor Whiskers"


class StartResponse(BaseModel):
    session_id: str
    labeled_count: int
    pool_remaining: int
    cv_accuracy: float
    cv_std: float


# ── POST /training/{session_id}/step ─────────────────────────────────────────

class MushroomFeatures(BaseModel):
    index: int
    features: dict


class PredictionResult(BaseModel):
    label: str           # "edible" | "poisonous"
    confidence: float
    p_edible: float
    p_poisonous: float
    dialogue: str


class StepResponse(BaseModel):
    mushroom: MushroomFeatures
    prediction: PredictionResult
    true_label: str      # "edible" | "poisonous"
    was_correct: bool
    new_accuracy: float
    cat_reaction: str    # "happy" | "sick" | "confused"
    labeled_count: int
    pool_remaining: int
    retrained: bool


# ── POST /training/{session_id}/strategy ──────────────────────────────────────

class StrategyRequest(BaseModel):
    strategy: str        # "random" | "uncertainty" | "committee" | "bald"


# ── GET /training/{session_id}/curve ─────────────────────────────────────────

class CurvePoint(BaseModel):
    labeled_count: int
    cv_accuracy: float
    cv_std: float


class LearningCurveResponse(BaseModel):
    session_id: str
    curve: list[CurvePoint]
