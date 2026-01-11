import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Chemins
    DATA_DIR = "data"
    VECTOR_DB_DIR = "chroma_db"
    
    # Embedding
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.1-8b-instant"  # ou "mixtral-8x7b-32768"
    
    # RAG Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 3
    
    @staticmethod
    def check_env():
        """Vérifie les variables d'environnement nécessaires"""
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY non trouvée dans .env")
        print("✓ Configuration chargée avec succès")
    
    @staticmethod
    def get_prompt_template():
        """Template du prompt pour le RAG"""
        return """CONTEXTE (extraits de documents):
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Réponds UNIQUEMENT en français
2. Base ta réponse UNIQUEMENT sur le contexte fourni ci-dessus
3. Si le contexte ne contient PAS l'information nécessaire, réponds: "Je ne trouve pas cette information dans les documents fournis"
4. Sois précis, concis et utile
5. Cite les sources quand c'est pertinent

RÉPONSE:"""