from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import Config

class EmbeddingService:
    """Service pour générer les embeddings"""
    
    def __init__(self):
        print(f"Chargement du modèle d'embedding: {Config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print("✓ Modèle d'embedding chargé")
    
    def embed_text(self, texts):
        """
        Génère des embeddings pour une liste de textes
        
        Args:
            texts: List[str] - Liste de textes à vectoriser
        
        Returns:
            np.ndarray - Matrice d'embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def get_embedding_dimension(self):
        """Retourne la dimension des embeddings"""
        # Test avec un texte court
        test_embedding = self.embed_text("test")
        return test_embedding.shape[1]