# SmartHelp - Micro-service de Support Multimodal & RAG Hybride

Bienvenue sur le dépôt du projet **SmartHelp**, un micro-service intelligent d'ingestion et de traitement de tickets de support client. Ce système combine l'intelligence artificielle générative, le traitement du langage naturel (NLP), la vision par ordinateur et l'analyse audio.

 **Lien du Kanban du Projet :** [Kanban Support Ticket API](https://github.com/users/Kya127/projects/1/views/2)

---

## Fonctionnalités Principales

- **Transcription Audio (ASR) :** Utilisation du modèle **Whisper** pour transcrire automatiquement les réclamations vocales des clients.
- **Analyse Visuelle (Computer Vision) :** Utilisation du modèle **BLIP** pour analyser les photos de produits endommagés ou non conformes.
- **Moteur RAG Hybride :** Combinaison d'une recherche vectorielle (ChromaDB + Embeddings HuggingFace `all-MiniLM-L6-v2`) et d'une recherche par mots-clés (**BM25**) avec un ensemble retriever, piloté par le LLM **Llama 3.3 (70B)** via Groq.
- **Orchestration FastAPI :** Une API moderne, rapide et documentée pour centraliser les requêtes multimodales.

---

##  Installation et Configuration

### 1. Cloner le projet
```bash
git clone [https://github.com/Kya127/micro-service-support.git](https://github.com/Kya127/micro-service-support.git)
cd micro-service-support

1. Créer et activer l'environnement virtuel

python -m venv venv
# Sur Windows (PowerShell) :
.env\Scripts\Activate.ps1

2. Installer les dépendances

pip install -r requirements.txt

3. Configurer les variables d'environnement

GROQ_API_KEY=ta_vraie_cle_api_groq


Démarrage de l'Application
Lance le serveur Uvicorn avec rechargement automatique :

uvicorn app.main:app --reload

L'application sera accessible localement :

Documentation interactive (Swagger UI) : http://127.0.0.1:8000/docs

Documentation alternative (ReDoc) : http://127.0.0.1:8000/redoc


Structure du Projet
Micro-service-suport/
│
├── app/
│   ├── routers/        # Routeurs FastAPI (support, audio, image, rag)
│   ├── services/       # Logique métier (asr_service, vision_service, rag_service)
│   └── main.py         # Point d'entrée de l'application FastAPI
│
├── .env                # Variables d'environnement (confidentiel)
├── requirements.txt    # Dépendances du projet
└── README.md           # Documentation du projet



Auteur
Projet réalisé par Maryam Harouna dans le cadre du cursus de formation chez Simplon Sénégal.