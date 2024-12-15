from pathlib import Path
import uuid
from dataclasses import dataclass
from typing import List, Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP

@dataclass
class ProcessedDocument:
    """Represents a processed document with its content and metadata"""
    id: str
    title: str
    content: str
    chunks: List[str]
    total_pages: int
    source_path: Path

class DocumentProcessor:
    """Handles the processing of PDF documents for content extraction and chunking"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
    
    def process_pdf(self, file_path: Path) -> ProcessedDocument:
        try:
            reader = PdfReader(str(file_path))
            
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            chunks = self.text_splitter.split_text(text)
            
            doc = ProcessedDocument(
                id=str(uuid.uuid4()),
                title=file_path.stem,
                content=text,
                chunks=chunks,
                total_pages=len(reader.pages),
                source_path=file_path
            )
            
            return doc
            
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
    
    def get_document_info(self, doc: ProcessedDocument) -> dict:
        return {
            "id": doc.id,
            "title": doc.title,
            "total_pages": doc.total_pages,
            "total_chunks": len(doc.chunks),
            "average_chunk_size": sum(len(chunk) for chunk in doc.chunks) / len(doc.chunks)
        }