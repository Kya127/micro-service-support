from fastapi import FastAPI
from app.routers import audio , image , rag
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(
    title="Micro-service de Support Client Multimodal",
    version="1.0.0"
)

# Ingestion des routeurs
app.include_router(audio.router)
app.include_router(image.router)
app.include_router(rag.router)

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "Le micro-service fonctionne correctement"}