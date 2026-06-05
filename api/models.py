from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import datetime
from api.database import Base

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, default="Default Project")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    pos_json = Column(Text, default="[]") 
    psos_json = Column(Text, default="[]")
    peos_json = Column(Text, default="[]")
    syllabi = relationship("SyllabusRecord", back_populates="project")

class SyllabusRecord(Base):
    __tablename__ = "syllabi"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    level = Column(String)
    programme = Column(String)
    course = Column(String)
    unit_titles = Column(Text)
    outcomes_text = Column(Text)
    cos_json = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    project = relationship("Project", back_populates="syllabi")
