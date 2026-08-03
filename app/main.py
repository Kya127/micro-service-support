from fastapi import FastAPI

app = FastAPI(
    title="API Support Client - Diagnostic Multimodal",
    description="Micro-service d'analyse automatique de réclamations avec Whisper, ViT et RAG",
    version="1.0.0"
)

@app.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "Le micro-service fonctionne correctement"}