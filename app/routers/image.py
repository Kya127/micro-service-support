from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.services.vision_service import get_vision_service

router = APIRouter(prefix="/image", tags=["Vision - Analyse d'Images"])

# Formats d'images autorisés
ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_image(file: UploadFile = File(...)):
    """Endpoint pour envoyer une image de défaut ou réclamation et obtenir une description."""
    
    # Validation du format d'extension
    if not file.filename.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Format d'image non supporté. Formats acceptés : {ALLOWED_IMAGE_EXTENSIONS}"
        )
    
    # Lecture des octets de l'image
    image_bytes = await file.read()
    
    # Appel du service Singleton
    vision_service = get_vision_service()
    description = vision_service.analyze_image(image_bytes)
    
    return {
        "filename": file.filename,
        "description": description
    }