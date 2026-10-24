from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Importações relativas corrigidas:
from .database import Base, engine, get_db
from .models import ClassificationHistory
from .schemas import PredictionInput, PredictionOutput, HistoryItem
from .ml_service import ml_service

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Flower Classifier API",
    version="1.0.0",
    description="API REST para classificação de flores Iris e gestão do histórico"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Flower Classifier API rodando normalmente!"}


@app.post("/api/v1/classify", response_model=PredictionOutput)
def classify_flower(data: PredictionInput, db: Session = Depends(get_db)):
    try:
        species, confidence, probas = ml_service.predict(
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        )

        db_entry = ClassificationHistory(
            sepal_length=data.sepal_length,
            sepal_width=data.sepal_width,
            petal_length=data.petal_length,
            petal_width=data.petal_width,
            predicted_species=species,
            confidence=confidence
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        return PredictionOutput(
            id=db_entry.id,
            predicted_species=species,
            confidence=confidence,
            probabilities=probas,
            created_at=db_entry.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/history", response_model=list[HistoryItem])
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    return db.query(ClassificationHistory).order_by(ClassificationHistory.created_at.desc()).limit(limit).all()
