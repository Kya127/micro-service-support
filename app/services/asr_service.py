import torch
from transformers import pipeline

class WhisperService:
    _instance = None

    def __new__(cls):
        # Pattern Singleton : instanciation unique
        if cls._instance is None:
            print("Chargement du modèle Whisper en mémoire...")
            cls._instance = super(WhisperService, cls).__new__(cls)
            
            # Utilisation du GPU si disponible, sinon CPU
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # On utilise whisper-tiny ou whisper-base pour des inférences rapides en local
            cls._instance.pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device
            )
            print("✅ Modèle Whisper prêt !")
        return cls._instance

    def transcribe(self, audio_bytes: bytes) -> str:
        """Reçoit les octets du fichier audio et renvoie le texte transcrit."""
        result = self.pipe(audio_bytes)
        return result.get("text", "").strip()


def get_whisper_service() -> WhisperService:
    return WhisperService()