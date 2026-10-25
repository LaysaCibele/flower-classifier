import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from .ml_service import ml_service

app = FastAPI(title="Flower Classifier API", version="1.0.0")

# Permite chamadas do frontend na Vercel e chamadas locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o cliente do Gemini
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


@app.post("/api/v1/classify")
@app.post("/predict")
def classify_flower(features: FlowerFeatures):
    try:
        predicted_species, confidence, probabilities = ml_service.predict(
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        )

        # Converte para float e ajusta a escala se o modelo retornar decimal (ex: 0.89 -> 89.0)
        conf_value = float(confidence)
        if conf_value <= 1.0 and conf_value > 0:
            conf_value = conf_value * 100.0

        conf_rounded = round(conf_value, 2)

        return {
            "predicted_species": predicted_species,
            "species": predicted_species,
            "confidence": conf_rounded,
            "probability": conf_rounded,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/api/v1/chat")
def chat_with_flora(chat: ChatMessage):
    if not gemini_client:
        msg = "A chave GEMINI_API_KEY não foi encontrada nas variáveis do Render."
        return {"response": msg, "reply": msg, "message": msg}

    try:
        prompt = (
            "Tu és a Flora 🌸, uma assistente virtual especialista no conjunto de dados Iris e botânica. "
            "Responde de forma simpática, clara e em português às dúvidas do utilizador.\n\n"
            f"Utilizador: {chat.message}"
        )
        # Usando o modelo estável gemini-1.5-flash
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )

        reply_text = response.text if (
            response and response.text) else "Desculpa, não consegui gerar uma resposta."

        return {
            "response": reply_text,
            "reply": reply_text,
            "message": reply_text,
            "data": {"response": reply_text}
        }
    except Exception as e:
        error_msg = f"Erro ao comunicar com a Flora: {str(e)}"
        return {"response": error_msg, "reply": error_msg, "message": error_msg}
