from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import RAGService

# 1. Création du routeur (et non d'une instance FastAPI)
router = APIRouter(prefix="/rag", tags=["RAG SmartHelp"])

class QuestionRequest(BaseModel):
    question: str

# Instance globale du service RAG
rag_service = RAGService()

# 2. Utilisation de @router.on_event au lieu de @app.on_event
@router.on_event("startup")
def startup_event():
    # Initialisation unique du service RAG au démarrage
    rag_service.initialize()

# 3. Utilisation de @router.post au lieu de @app.post
@router.post("/chat")
def chat_endpoint(request: QuestionRequest):
    try:
        response_text = rag_service.query(request.question)
        return {
            "question": request.question,
            "response": response_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))