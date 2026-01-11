import os
import tempfile
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

class DocumentProcessor:
    """Traitement des documents (PDF, TXT, DOCX)"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_uploaded_file(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """
        Traite un fichier uploadé et le découpe en chunks
        
        Args:
            file_path: Chemin du fichier
            filename: Nom original du fichier
        
        Returns:
            Liste de chunks avec métadonnées
        """
        # Extraction du texte selon le type de fichier
        text = self._extract_text(file_path, filename)
        
        if not text.strip():
            raise ValueError(f"Fichier vide ou impossible à lire: {filename}")
        
        # Découpage en chunks
        chunks = self.text_splitter.split_text(text)
        
        # Préparation des documents pour la base vectorielle
        documents = []
        for i, chunk in enumerate(chunks):
            document = {
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": os.path.splitext(filename)[1].lower(),
                }
            }
            documents.append(document)
        
        print(f"✓ Fichier '{filename}' traité: {len(chunks)} chunks créés")
        return documents
    
    def _extract_text(self, file_path: str, filename: str) -> str:
        """Extrait le texte d'un fichier selon son type"""
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.pdf':
            return self._extract_pdf(file_path)
        elif ext == '.txt':
            return self._extract_txt(file_path)
        elif ext in ['.docx', '.doc']:
            return self._extract_docx(file_path)
        else:
            raise ValueError(f"Format non supporté: {ext}")
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extrait le texte d'un PDF"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Erreur lecture PDF: {e}")
    
    def _extract_txt(self, file_path: str) -> str:
        """Extrait le texte d'un fichier TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Essayer d'autres encodages
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                raise Exception(f"Erreur lecture TXT: {e}")
    
    def _extract_docx(self, file_path: str) -> str:
        """Extrait le texte d'un fichier DOCX"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Erreur lecture DOCX: {e}")
    
    def process_multiple_files(self, files: List[Any]) -> List[Dict[str, Any]]:
        """
        Traite plusieurs fichiers uploadés
        
        Args:
            files: Liste de fichiers (objets gradio.File ou tuples (path, name))
        
        Returns:
            Liste combinée de tous les chunks
        """
        all_documents = []
        
        for file_info in files:
            if isinstance(file_info, tuple):
                file_path, filename = file_info
            else:
                file_path = file_info.name
                filename = os.path.basename(file_path)
            
            try:
                documents = self.process_uploaded_file(file_path, filename)
                all_documents.extend(documents)
            except Exception as e:
                print(f"⚠️ Erreur traitement {filename}: {e}")
                continue
        
        return all_documents