from pydantic import BaseModel, Field
from datetime import datetime


class PredictionInput(BaseModel):
    sepal_length: float = Field(..., gt=0,
                                description="Comprimento da sépala em cm")
    sepal_width: float = Field(..., gt=0,
                               description="Largura da sépala em cm")
    petal_length: float = Field(..., gt=0,
                                description="Comprimento da pétala em cm")
    petal_width: float = Field(..., gt=0,
                               description="Largura da pétala em cm")


class PredictionOutput(BaseModel):
    id: int
    predicted_species: str
    confidence: float
    probabilities: dict[str, float]
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    id: int
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    predicted_species: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True
