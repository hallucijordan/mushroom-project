from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from backend.app.db.database import Base

class TrainingSession(Base):
    __tablename__ = "training_session"
    id = Column(String, primary_key=True)
    cat_name = Column(String, default="Professor Whiskers")
    cat_level = Column(Integer, default=0)
    cat_xp = Column(Integer, default=0)
    learner_type = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    init_strategy = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

class LabeledSample(Base):
    __tablename__ = "labeled_sample"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("training_session.id"))
    sample_index = Column(Integer)
    true_label = Column(Integer)
    predicted_label = Column(Integer)
    predicted_conf = Column(Float)
    was_correct = Column(Boolean)
    labeled_at = Column(DateTime, server_default=func.now())