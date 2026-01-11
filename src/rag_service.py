import time
from typing import List, Dict, Any
from groq import Groq
from src.config import Config
from src.database import VectorDatabase

class RAGService:
    """Service principal RAG avec intégration Groq API"""
    
    def __init__(self):
        print("Initialisation du service RAG...")
        
        # 1. Initialiser la base vectorielle d'abord
        print(" - Initialisation base vectorielle...")
        self.vector_db = VectorDatabase()
        
        # 2. Initialiser Groq après
        print(" - Initialisation client Groq...")
        self.groq_client = self._init_groq_client()
        
        print("✓ Service RAG initialisé")
    
    def _init_groq_client(self):
        """Initialise le client Groq de manière isolée"""
        try:
            # Import ici pour isoler d'éventuels conflits
            from groq import Groq
            
            # Essayer différentes méthodes d'initialisation
            api_key = Config.GROQ_API_KEY
            
            if not api_key or api_key == "votre_cle_api_groq_ici":
                print("⚠️  Clé Groq API non configurée ou par défaut")
                return None
            
            print(f"   Clé API: {api_key[:10]}...")
            
            try:
                client = Groq(api_key=api_key)
                print("   ✓ Client Groq initialisé avec api_key")
                return client
            except TypeError as e:
                print(f"   Essai méthode alternative (erreur: {e})")
                # Essayer sans paramètre
                client = Groq()
                print("   ✓ Client Groq initialisé sans paramètre")
                return client
                
        except ImportError as e:
            print(f"✗ Impossible d'importer groq: {e}")
            return None
        except Exception as e:
            print(f"✗ Erreur initialisation Groq: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_and_store_documents(self, files: List[Any]) -> Dict[str, Any]:
        """
        Traite et stocke les documents uploadés
        
        Args:
            files: Liste de fichiers (depuis Gradio ou tuples)
        
        Returns:
            Dict avec statistiques
        """
        from src.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        documents = processor.process_multiple_files(files)
        
        if not documents:
            return {"success": False, "message": "Aucun document valide", "count": 0}
        
        count = self.vector_db.add_documents(documents)
        
        return {
            "success": True,
            "message": f"{count} documents traités et stockés",
            "count": count,
            "total_chunks": len(documents)
        }
    
    def generate_answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Génère une réponse à une question en utilisant le RAG
        
        Args:
            question: Question de l'utilisateur
            top_k: Nombre de contextes à récupérer
        
        Returns:
            Dict avec réponse et métadonnées
        """
        # 1. Recherche de contextes pertinents
        start_time = time.time()
        relevant_docs = self.vector_db.search(question, top_k)
        search_time = time.time() - start_time
        
        if not relevant_docs:
            return {
                "answer": "Je n'ai pas trouvé d'informations pertinentes dans les documents pour répondre à votre question.",
                "sources": [],
                "stats": {
                    "search_time": search_time,
                    "generation_time": 0,
                    "total_time": search_time,
                    "documents_used": 0
                }
            }
        
        # 2. Construction du contexte
        context = self._build_context(relevant_docs)
        
        # 3. Construction du prompt
        prompt = Config.get_prompt_template().format(
            context=context,
            question=question
        )
        
        # 4. Génération avec Groq
        start_gen = time.time()
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un assistant utile qui répond aux questions en se basant strictement sur le contexte fourni."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=Config.GROQ_MODEL,
                temperature=0.1,
                max_tokens=500,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            generation_time = time.time() - start_gen
            
        except Exception as e:
            answer = f"Erreur lors de la génération: {str(e)}"
            generation_time = time.time() - start_gen
        
        # 5. Formatage des sources
        sources = [
            {
                "content": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                "source": doc["metadata"].get("source", "Inconnu"),
                "score": round(doc["score"], 3),
                "chunk": doc["metadata"].get("chunk_index", 0) + 1
            }
            for doc in relevant_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "stats": {
                "search_time": round(search_time, 2),
                "generation_time": round(generation_time, 2),
                "total_time": round(search_time + generation_time, 2),
                "documents_used": len(relevant_docs)
            }
        }
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Construit le contexte à partir des documents pertinents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc["metadata"].get("source", f"Document {i}")
            chunk_num = doc["metadata"].get("chunk_index", 0) + 1
            context_parts.append(f"[Source: {source}, Chunk {chunk_num}]:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le système"""
        stats = self.vector_db.get_collection_stats()
        return {
            "vector_db": stats,
            "groq_model": Config.GROQ_MODEL,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "top_k": Config.TOP_K_RESULTS
        }
    
    def reset_database(self):
        """Réinitialise la base de données"""
        self.vector_db.reset_collection()
        return {"success": True, "message": "Base de données réinitialisée"}