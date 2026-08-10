from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import json

# Base de connaissances SmartHelp intégrée directement
SMARTHELP_POLICY = """
BASE DE CONNAISSANCES INTERNE - POLITIQUE DE SUPPORT ET DE RETOUR (SMARTHELP)

SECTION 1 : PRODUITS ENDOMMAGÉS OU CASSÉS À LA LIVRAISON
- Règle 1.1 (Casse / Dommage visible) : Si le client signale un produit cassé, fissuré ou endommagé, et fournit une photo probante de l'article dans un délai de 48 heures suivant la réception, le dossier est éligible à un remboursement intégral ou à un renvoi gratuit. Statut associé : "Remboursable".
- Règle 1.2 (Délai dépassé) : Si la réclamation pour produit cassé est faite après un délai de 48 heures, la demande est soumise à validation manuelle du manager. Statut associé : "À vérifier".

SECTION 2 : ERREURS DE LIVRAISON ET NON-CONFORMITÉ DE COMMANDE
- Règle 2.1 (Mauvais article reçu) : Si le produit reçu ne correspond pas en termes de modèle, de couleur ou de taille par rapport à la commande initiale, l'échange est entièrement pris en charge par l'entreprise (frais de retour offerts). Statut associé : "Échange gratuit".
- Règle 2.2 (Pièce manquante) : Si un accessoire ou une pièce d'un kit est manquant à la réception, l'entreprise s'engage à expédier uniquement la pièce manquante sous 3 jours ouvrés. Statut associé : "Expédition de pièce".

SECTION 3 : RETARDS DE LIVRAISON ET PROBLÈMES LOGISTIQUES
- Règle 3.1 (Retard mineur) : Un retard de livraison inférieur ou égal à 3 jours ouvrés par rapport à la date estimée ne donne droit à aucun dédommagement financier. Statut associé : "Non remboursable - Retard mineur".
- Règle 3.2 (Retard majeur) : Un retard supérieur à 5 jours ouvrés donne droit à un bon d'achat de 10% valable sur la prochaine commande. Statut associé : "Dédommagement 10%".
- Règle 3.3 (Colis perdu) : Si le statut du transporteur indique "Bloqué" ou "Perdu" depuis plus de 7 jours, un remboursement total est déclenché automatiquement après enquête transporteur. Statut associé : "Remboursable - Colis perdu".

SECTION 4 : RÈGLES GÉNÉRALES DE REFUS (HORS CADRE)
- Règle 4.1 (Usure normale / Mauvaise utilisation) : Si l'analyse visuelle ou la description audio montre que le défaut est dû à une mauvaise manipulation du client, à une chute après réception ou à une usure normale, la réclamation est rejetée. Statut associé : "Refusé".
- Règle 4.2 (Absence de preuve) : Toute réclamation concernant un produit endommagé ou non conforme qui ne comporte ni photo probante ni description claire par message vocal sera mise en attente de pièces complémentaires. Statut associé : "En attente de justificatifs".
"""

class RAGService:
    _instance = None

    def __new__(cls, model_name: str = "gpt-4o-mini"):
        # Pattern Singleton : s'assure qu'une seule instance existe en mémoire
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.rag_chain = None
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        print("\n⏳ Initialisation du moteur RAG Hybride SmartHelp...")

        policy_document = Document(
            page_content=SMARTHELP_POLICY,
            metadata={"source": "politique_smarthelp.txt"}
        )
        documents = [policy_document]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 3

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )

        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        prompt_template = """Tu es une assistante expert spécialisée dans la politique de support et de retour SmartHelp.
Tu dois répondre aux questions en te basant UNIQUEMENT sur le contexte fourni ci-dessous.

Règles de réponse :
1. Si l'utilisateur salue ou entame la conversation, accueille-le poliment.
2. Si l'utilisateur te remercie ou clôture la conversation (ex: "merci", "d'accord", "parfait"), réponds simplement par une formule de politesse brève (ex: "Avec plaisir, n'hésitez pas si vous avez d'autres questions !, A bientot!"), SANS redemander de détails ni répéter un message d'accueil.
3. Lorsqu'un cas client ou une réclamation est décrit, identifie précisément la règle applicable et mentionne TOUJOURS le statut associé après avoir fait un bref résumé de la situation.
4. Si la réponse n'est pas présente dans le contexte, indique poliment que l'information n'est pas disponible dans le règlement intérieur.

IMPORTANT — Format de sortie obligatoire :
Tu dois TOUJOURS répondre avec un objet JSON valide, et rien d'autre autour (pas de texte avant ou après, pas de balises markdown ```json) hormis les messages de salutations ou de remerciements.
Structure attendue :

{{
  "reponse": "<ta réponse habituelle, rédigée pour le client, en français>",
  "analyse": {{
    "Type de dommage": "<type de dommage identifié, ou néant si non applicable>",
    "confiance": <un nombre entre 0 et 100 représentant ta confiance dans le diagnostic>,
    "Dégat": "<Faible, Moyenne ou Élevée, ou néant si non applicable>",
    "Recommandation": "<action recommandée en une courte phrase, ou null si non applicable>"
  }}
}}

Si la situation ne concerne pas une réclamation avec un dommage identifiable (ex: simple salutation, question générale), mets "analyse": null.

Contexte:
{context}

Question:
{question}

Réponse (JSON uniquement):"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("Moteur RAG Hybride SmartHelp prêt !\n")

    def parse_structured_response(self, raw_text: str) -> dict:
        """Transforme le texte JSON renvoyé par le modèle en dictionnaire Python.
        Si jamais le modèle ne respecte pas le format, on retombe sur une
        réponse simple en texte, sans planter l'application."""
        try:
            cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(cleaned)
            return {
                "response": parsed.get("reponse", raw_text),
                "analysis": parsed.get("analyse", None)
            }
        except (json.JSONDecodeError, AttributeError):
            return {"response": raw_text, "analysis": None}

    def query(self, question: str) -> dict:
        if self.rag_chain is None:
            raise RuntimeError("Le service RAG n'est pas encore initialisé.")
        raw_text = self.rag_chain.invoke(question)
        return self.parse_structured_response(raw_text)