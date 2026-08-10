from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.asr_service import get_whisper_service

router = APIRouter(prefix="/audio", tags=["Audio - ASR"])

# Formats audio acceptés
ALLOWED_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mpeg")

@router.post("/transcribe", status_code=status.HTTP_200_OK)
async def transcribe_audio(file: UploadFile = File(...)):
    """Endpoint pour envoyer un fichier audio et obtenir sa transcription texte."""
    
    # Validation de l'extension du fichier
    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400, 
            detail=f"Format non supporté. Formats acceptés : {ALLOWED_EXTENSIONS}"
        )
    
    # Lecture du fichier audio envoyé
    audio_bytes = await file.read()
    
    # Inférence via notre service Singleton
    asr_service = get_whisper_service()
    transcription = asr_service.transcribe(audio_bytes)
    
    return {
        "filename": file.filename,
        "transcription": transcription
    }