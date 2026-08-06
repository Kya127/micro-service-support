from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.rag_service import RAGService
from app.services.asr_service import get_whisper_service
from app.services.vision_service import get_vision_service

router = APIRouter(prefix="/support", tags=["Orchestration Support Multimodal"])

# Initialisation unique des services (bénéficient du pattern Singleton)
rag_service = RAGService()
whisper_service = get_whisper_service()
vision_service = get_vision_service()

@router.post("/support-ticket")
async def process_support_ticket(
    text: str = Form(None, description="Message textuel du client"),
    audio: UploadFile = File(None, description="Fichier audio de la réclamation (.wav, .mp3)"),
    image: UploadFile = File(None, description="Photo du produit endommagé (.jpg, .png)")
):
    extracted_text_from_audio = ""
    image_description = ""

    try:
        # 1. Traitement audio : lecture directe en mémoire des octets
        if audio:
            audio_bytes = await audio.read()
            extracted_text_from_audio = whisper_service.transcribe(audio_bytes)

        # 2. Traitement image : lecture directe en mémoire des octets
        if image:
            image_bytes = await image.read()
            image_description = vision_service.analyze_image(image_bytes)

        # 3. Consolidation du contexte global pour le RAG
        combined_query = (
            f"Message texte : {text or 'Aucun'}. \n"
            f"Transcription audio : {extracted_text_from_audio}. \n"
            f"Analyse visuelle : {image_description}."
        )

        # 4. Interrogation du moteur RAG SmartHelp (avec Groq)
        rag_response = rag_service.query(combined_query)

        # 5. Réponse JSON unifiée
        return {
            "status": "success",
            "inputs_recus": {
                "texte": text,
                "audio_nom": audio.filename if audio else None,
                "image_nom": image.filename if image else None
            },
            "traitements_multimodaux": {
                "transcription_audio": extracted_text_from_audio,
                "diagnostic_visuel": image_description
            },
            "synthese_rag": rag_response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement multimodal du ticket : {str(e)}"
        )