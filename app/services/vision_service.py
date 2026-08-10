import io
import torch
from PIL import Image
from transformers import pipeline

class VisionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("⏳ Chargement du modèle de Vision (BLIP) en mémoire...")
            cls._instance = super(VisionService, cls).__new__(cls)
            
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            cls._instance.pipe = pipeline(
                "image-text-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=device
            )
            print("✅ Modèle Vision prêt !")
        return cls._instance

    def analyze_image(self, image_bytes: bytes) -> str:
        """Décode l'image en mémoire et génère sa description."""
        try:
            # 1. Chargement de l'image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            default_prompt = "a detailed description of"
            # 2. Inférence du modèle
            result = self.pipe(image , text=default_prompt)
            print(f"Output brut du modèle BLIP : {result}")
            
            # 3. Extraction sécurisée du texte
            if result and isinstance(result, list) and len(result) > 0:
                first_res = result[0]
                if isinstance(first_res, dict):
                    # Cherche 'generated_text' ou renvoie la valeur disponible
                    text = first_res.get("generated_text") or first_res.get("caption") or str(first_res)
                    if isinstance(text, list) and len(text) > 0:
                        text = text[0].get("generated_text", str(text[0]))
                    return str(text).strip()
            
            return "Aucune description générée."

        except Exception as e:
            print(f"❌ Erreur pendant l'analyse d'image : {e}")
            raise e


def get_vision_service() -> VisionService:
    return VisionService()