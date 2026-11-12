"""Pydantic schemas for the /api/detect endpoints."""
from pydantic import BaseModel


class JudgeAResult(BaseModel):
    # ML classifier result after Gemini feature extraction
    label: str        # "edible" | "poisonous"
    confidence: float
    p_edible: float
    p_poisonous: float
    features: dict    # the 20 features extracted by Gemini


class JudgeBResult(BaseModel):
    # Gemini direct vision verdict
    verdict: str      # "edible" | "poisonous"
    confidence: float
    narrative: str


class ConsensusResult(BaseModel):
    verdict: str      # final weighted verdict
    confidence: float
    agreement: bool   # True if both judges agree


class AnalyzeResponse(BaseModel):
    judge_a: JudgeAResult
    judge_b: JudgeBResult
    consensus: ConsensusResult
