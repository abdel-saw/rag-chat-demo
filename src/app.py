#!/usr/bin/env python3
"""
Interface Gradio pour l'application RAG - Version corrigée
"""

import gradio as gr
import os
import tempfile
from typing import List
from src.config import Config
from src.rag_service import RAGService

class RAGGradioApp:
    """Application Gradio pour le système RAG"""
    
    def __init__(self):
        self.rag_service = RAGService()
        self.setup_interface()
    
    def setup_interface(self):
        """Configure l'interface Gradio"""
        
        with gr.Blocks(title="RAG Chat Demo", theme=gr.themes.Soft()) as self.app:
            gr.Markdown("# 🤖 RAG Chat - Démonstration")
            gr.Markdown("""
            **Téléchargez des documents (PDF, TXT, DOCX) et posez des questions !**
            
            Le système recherchera dans vos documents pour fournir des réponses précises.
            """)
            
            with gr.Tabs():
                # Onglet 1: Upload de documents
                with gr.Tab("📁 Upload Documents"):
                    gr.Markdown("### Téléchargez vos documents")
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_input = gr.File(
                                label="Documents",
                                file_types=[".pdf", ".txt", ".docx"],
                                file_count="multiple"
                            )
                            with gr.Row():
                                upload_btn = gr.Button("📤 Vectoriser et Stocker", variant="primary")
                                reset_btn = gr.Button("🔄 Réinitialiser Base", variant="secondary")
                            
                            status_output = gr.Textbox(
                                label="Status",
                                interactive=False,
                                placeholder="En attente de documents..."
                            )
                        
                        with gr.Column(scale=1):
                            stats_box = gr.JSON(
                                label="📊 Statistiques Système",
                                value=self.rag_service.get_system_info()
                            )
                
                # Onglet 2: Chat RAG
                with gr.Tab("💬 Chat RAG"):
                    gr.Markdown("### Posez vos questions sur les documents")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400
                    )
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Votre question",
                            placeholder="Posez une question sur vos documents...",
                            scale=4
                        )
                        submit_btn = gr.Button("Envoyer", variant="primary", scale=1)
                    
                    with gr.Accordion("⚙️ Paramètres", open=False):
                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=Config.TOP_K_RESULTS,
                                step=1,
                                label="Nombre de contextes (top-k)"
                            )
                    
                    with gr.Accordion("📚 Sources Utilisées", open=False):
                        sources_output = gr.JSON(label="Sources")
                    
                    with gr.Accordion("📊 Métriques", open=False):
                        metrics_output = gr.JSON(label="Performances")
            
            # Événements
            upload_btn.click(
                fn=self.process_documents,
                inputs=[file_input],
                outputs=[status_output, stats_box]
            )
            
            reset_btn.click(
                fn=self.reset_database,
                inputs=[],
                outputs=[status_output, stats_box]
            )
            
            submit_btn.click(
                fn=self.ask_question,
                inputs=[question_input, chatbot, top_k_slider],
                outputs=[chatbot, question_input, sources_output, metrics_output]
            )
            
            question_input.submit(
                fn=self.ask_question,
                inputs=[question_input, chatbot, top_k_slider],
                outputs=[chatbot, question_input, sources_output, metrics_output]
            )
    
    def process_documents(self, files: List[tempfile._TemporaryFileWrapper]):
        """Traite les documents uploadés"""
        if not files:
            return "❌ Aucun fichier sélectionné", self.rag_service.get_system_info()
        
        try:
            result = self.rag_service.process_and_store_documents(files)
            
            if result["success"]:
                message = f"✅ {result['message']} ({result['total_chunks']} chunks)"
            else:
                message = f"❌ {result['message']}"
            
            return message, self.rag_service.get_system_info()
            
        except Exception as e:
            return f"❌ Erreur: {str(e)[:200]}", self.rag_service.get_system_info()
    
    def ask_question(self, question: str, chat_history, top_k: int):
        """Traite une question et retourne la réponse - FORMAT CORRIGÉ"""
        if not question.strip():
            return chat_history, "", {}, {}
        
        # Ajouter la question au format Gradio moderne
        chat_history.append({"role": "user", "content": question})
        
        try:
            # Générer la réponse
            response = self.rag_service.generate_answer(question, top_k)
            
            # Ajouter la réponse au format Gradio moderne
            chat_history.append({"role": "assistant", "content": response["answer"]})
            
            # Préparer les sources
            sources = response.get("sources", [])
            metrics = response.get("stats", {})
            
            return chat_history, "", sources, metrics
            
        except Exception as e:
            error_msg = f"❌ Erreur: {str(e)[:200]}"
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, "", {}, {}
    
    def reset_database(self):
        """Réinitialise la base de données"""
        try:
            result = self.rag_service.reset_database()
            return f"✅ {result['message']}", self.rag_service.get_system_info()
        except Exception as e:
            return f"❌ Erreur: {str(e)[:200]}", self.rag_service.get_system_info()
    
    def launch(self, share: bool = False, debug: bool = False):
        """Lance l'application"""
        print("\n" + "="*50)
        print("🚀 Lancement de l'application RAG Chat")
        print("="*50)
        print(f"📁 Données: {Config.DATA_DIR}")
        print(f"🔧 Base vectorielle: {Config.VECTOR_DB_DIR}")
        print(f"🧠 Modèle d'embedding: {Config.EMBEDDING_MODEL}")
        print(f"🤖 Modèle Groq: {Config.GROQ_MODEL}")
        print("="*50)
        print("\n📢 L'interface web va s'ouvrir...")
        print("👉 Accédez à: http://localhost:7860")
        print("👉 Appuyez sur Ctrl+C pour arrêter")
        
        self.app.launch(
            server_name="127.0.0.1",  # Changé pour localhost
            server_port=7860,
            share=share,
            debug=debug,
            show_error=True
        )

# Point d'entrée
def main():
    """Fonction principale"""
    # Vérifier la configuration
    Config.check_env()
    
    # Lancer l'application
    app = RAGGradioApp()
    app.launch(share=False, debug=True)

if __name__ == "__main__":
    main()