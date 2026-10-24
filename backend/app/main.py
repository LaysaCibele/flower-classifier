import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from .ml_service import ml_service

app = FastAPI(title="Flower Classifier API", version="1.0.0")

# Permitir CORS para a Vercel e chamadas locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cliente do Gemini
gemini_client = None
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    gemini_client = genai.Client(api_key=api_key)


class FlowerFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class ChatMessage(BaseModel):
    message: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Rota de classificação esperada pelo Frontend
@app.post("/api/v1/classify")
@app.post("/predict")  # Mantém /predict para retrocompatibilidade
def classify_flower(features: FlowerFeatures):
    try:
        predicted_species, confidence, probabilities = ml_service.predict(
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        )
        return {
            "predicted_species": predicted_species,
            "confidence": confidence,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro na predição: {str(e)}")


# Rota do chat da Flora esperada pelo Frontend
@app.post("/api/v1/chat")
def chat_with_flora(chat: ChatMessage):
    if not gemini_client:
        raise HTTPException(
            status_code=500, detail="GEMINI_API_KEY não configurada no servidor.")

    try:
        prompt = (
            "Tu és a Flora 🌸, uma assistente virtual especialista no conjunto de dados Iris e botânica. "
            "Responde de forma simpática, clara e em português às dúvidas do utilizador.\n\n"
            f"Utilizador: {chat.message}"
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro ao comunicar com o Gemini: {str(e)}")
