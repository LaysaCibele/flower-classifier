import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .ml_service import ml_service
from .agent_service import ask_gemini, build_flora_prompt, AGENT_VERSION, _discover_flash_models, PREFERRED_MODELS

app = FastAPI(title="Flower Classifier API", version="1.0.0")

# Permitir CORS para a Vercel e chamadas locais
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FlowerFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1)


@app.get("/")
def root():
    return {"service": "flower-classifier", "status": "ok"}


@app.get("/debug/gemini")
def debug_gemini():
    """Mostra qual versão do código está no ar e quais modelos a chave enxerga (sem gastar cota)."""
    from google import genai
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    info = {"agent_version": AGENT_VERSION, "api_key_set": bool(
        api_key), "preferred_models": [x for x in PREFERRED_MODELS if x]}
    if api_key:
        try:
            info["models_visible_to_key"] = _discover_flash_models(
                genai.Client(api_key=api_key))
        except Exception as e:
            info["list_error"] = str(e)
    return info


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
    try:
        prompt = build_flora_prompt(user_message=chat.message)
        reply_text = ask_gemini(prompt, user_message=chat.message)

        return {
            "response": reply_text,
            "reply": reply_text,
            "message": reply_text,
            "data": {"response": reply_text}
        }
    except Exception as e:
        error_msg = f"Erro ao comunicar com a Flora: {str(e)}"
        return {"response": error_msg, "reply": error_msg, "message": error_msg}
