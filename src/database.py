import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid
from src.config import Config
from src.embeddings import EmbeddingService

class VectorDatabase:
    """Gestion de la base de données vectorielle ChromaDB"""
    
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()
        self.collection = self._get_or_create_collection()
        print(f"✓ Base vectorielle initialisée: {Config.VECTOR_DB_DIR}")
    
    def _get_or_create_collection(self):
        """Récupère ou crée la collection ChromaDB"""
        try:
            # Essayer de récupérer la collection existante
            collection = self.client.get_collection(self.collection_name)
            print(f"Collection existante chargée: {self.collection_name}")
            return collection
        except:
            # Créer une nouvelle collection avec embedding_function personnalisée
            print(f"Création nouvelle collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self._get_embedding_function()
            )
    
    def _get_embedding_function(self):
        """Retourne une fonction d'embedding compatible avec ChromaDB"""
        from chromadb import EmbeddingFunction
        
        class CustomEmbeddingFunction(EmbeddingFunction):
            def __init__(self, embedding_service):
                self.embedding_service = embedding_service
            
            def __call__(self, input: List[str]) -> List[List[float]]:
                """Nouvelle signature requise par ChromaDB"""
                embeddings = self.embedding_service.embed_text(input)
                return embeddings.tolist()
        
        return CustomEmbeddingFunction(self.embedding_service)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Ajoute des documents à la base vectorielle
        
        Args:
            documents: Liste de dicts avec 'text', 'metadata', 'id' (optionnel)
        
        Returns:
            int: Nombre de documents ajoutés
        """
        if not documents:
            return 0
        
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        ids = [doc.get("id", str(uuid.uuid4())) for doc in documents]
        
        # Ajout à ChromaDB
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ {len(documents)} documents ajoutés à la base vectorielle")
        return len(documents)
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Recherche les documents les plus similaires à la requête
        
        Args:
            query: Texte de la requête
            top_k: Nombre de résultats (par défaut: Config.TOP_K_RESULTS)
        
        Returns:
            Liste de documents avec score de similarité
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formatage des résultats
        documents = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - (results["distances"][0][i] if results["distances"] else 0),
                    "id": results["ids"][0][i] if results["ids"] else None
                }
                documents.append(doc)
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la collection"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_dimension": self.embedding_service.get_embedding_dimension()
        }
    
    def reset_collection(self):
        """Réinitialise la collection (utile pour les tests)"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection {self.collection_name} supprimée")
        except:
            pass
        self.collection = self._get_or_create_collection()
        print("Collection réinitialisée")