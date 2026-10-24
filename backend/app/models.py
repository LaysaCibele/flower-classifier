from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from .database import Base 


class ClassificationHistory(Base):
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_species = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
