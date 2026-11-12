"""Session router — /api/session"""
from fastapi import APIRouter, HTTPException
from backend.app.dependencies import al_engine
from backend.app.schemas.training import StartResponse, LearningCurveResponse, CurvePoint

router = APIRouter()


@router.get("/{session_id}", response_model=StartResponse)
async def get_session(session_id: str):
    if session_id not in al_engine.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    state = al_engine.get_state(session_id)
    return StartResponse(session_id=session_id, **state)


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
