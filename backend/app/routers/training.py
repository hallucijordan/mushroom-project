from fastapi import APIRouter, HTTPException
from backend.app.dependencies import al_engine
from backend.app.schemas.training import (
    StartRequest, StartResponse,
    StepResponse,
    StrategyRequest,
    LearningCurveResponse, CurvePoint,
)

router = APIRouter()


@router.post("/start", response_model=StartResponse)
async def start_training(req: StartRequest):
    session_id = al_engine.create_session(
        learner_key=req.learner_type,
        strategy_key=req.strategy,
        init_method=req.init_strategy,
    )
    state = al_engine.get_state(session_id)
    return StartResponse(session_id=session_id, **state)


@router.post("/{session_id}/step", response_model=StepResponse)
async def train_step(session_id: str):
    """
    One atomic auto-training step: query → predict → auto-reveal → maybe retrain.
    The frontend calls this in a loop to drive the training animation.
    """
    if session_id not in al_engine.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    result = al_engine.train_step(session_id)
    return result


@router.post("/{session_id}/strategy")
async def set_strategy(session_id: str, req: StrategyRequest):
    """Change the active query strategy mid-session."""
    if session_id not in al_engine.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    al_engine.set_strategy(session_id, req.strategy)
    return {"session_id": session_id, "strategy": req.strategy}


@router.get("/{session_id}/curve", response_model=LearningCurveResponse)
async def get_curve(session_id: str):
    if session_id not in al_engine.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = al_engine.sessions[session_id]
    curve = [CurvePoint(**p) for p in session.get_learning_curve()]
    return LearningCurveResponse(session_id=session_id, curve=curve)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    if session_id not in al_engine.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del al_engine.sessions[session_id]
    return {"deleted": session_id}
