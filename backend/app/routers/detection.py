"""Detection router — /api/detect"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from backend.app.dependencies import al_engine, gemini
from backend.app.schemas.detection import AnalyzeResponse, JudgeAResult, JudgeBResult, ConsensusResult

router = APIRouter()
_al_engine = al_engine
_gemini = gemini


def _compute_consensus(judge_a: dict, judge_b: dict) -> ConsensusResult:
    """Weighted consensus: Judge A (ML) gets 0.6 weight, Judge B (vision) 0.4."""
    p_poisonous_a = judge_a["p_poisonous"]
    # Convert judge B confidence to p_poisonous
    p_poisonous_b = judge_b["confidence"] if judge_b["verdict"] == "poisonous" else 1 - judge_b["confidence"]

    combined = 0.6 * p_poisonous_a + 0.4 * p_poisonous_b
    verdict = "poisonous" if combined >= 0.5 else "edible"
    agreement = judge_a["label"] == judge_b["verdict"]

    return ConsensusResult(verdict=verdict, confidence=combined if verdict == "poisonous" else 1 - combined, agreement=agreement)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)):
    """
    Upload a mushroom photo and get dual-judge analysis.

    Judge A: Gemini extracts features → ML classifier predicts
    Judge B: Gemini makes a direct visual verdict
    """
    img = Image.open(image.file).convert("RGB")

    # Judge A: feature extraction → ML
    features = _gemini.extract_features(img)
    if not features:
        raise HTTPException(status_code=422, detail="Gemini could not extract features from this image.")

    # Use the default trained preprocessor + learner from ALEngine
    try:
        _al_engine._load_data()
        X = _al_engine._preprocessor.feature_dict_to_array(features)
        # Use whichever session's learner is available, or fall back to a fresh default
        if _al_engine.sessions:
            learner = next(iter(_al_engine.sessions.values())).learner
        else:
            from mushroom_ml import get_learner
            learner = get_learner("random_forest")
            learner.fit(_al_engine._X_full[:500], _al_engine._y_full[:500])
        prediction_a = learner.predict_single(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {e}")

    judge_a = JudgeAResult(
        label=prediction_a["label"],
        confidence=prediction_a["confidence"],
        p_edible=prediction_a["p_edible"],
        p_poisonous=prediction_a["p_poisonous"],
        features=features,
    )

    # Judge B: direct Gemini vision verdict
    result_b = _gemini.direct_judge(img)
    judge_b = JudgeBResult(**result_b)

    consensus = _compute_consensus(prediction_a, result_b)

    return AnalyzeResponse(judge_a=judge_a, judge_b=judge_b, consensus=consensus)
