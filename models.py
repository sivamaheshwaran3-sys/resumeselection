from sqlalchemy import Column, Integer, String, Float, Text
from database import Base

class JobDescription(Base):
    __tablename__ = "job_descriptions"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    text = Column(Text)

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    text = Column(Text)

class Score(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, index=True)
    resume_id = Column(Integer)
    jd_id = Column(Integer)
    hard_score = Column(Float)
    soft_score = Column(Float)
    hybrid_score = Column(Float)
    verdict = Column(String)
    missing_skills = Column(Text)
    suggestions = Column(Text)
