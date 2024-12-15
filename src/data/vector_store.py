from typing import List, Dict, Optional
from pathlib import Path
import faiss
import numpy as np
from langchain_community.embeddings import OpenAIEmbeddings
from src.utils.config import OPENAI_API_KEY, VECTOR_STORE_PATH
from src.core.document_processor import ProcessedDocument

class VectorStore:
    """Manages document embeddings and semantic search using FAISS"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.index = None
        self.document_map: Dict[int, str] = {}
        self.current_id = 0
    
    def add_document(self, document: ProcessedDocument) -> None:
        embeddings = []
        for chunk in document.chunks:
            vector = self.embeddings.embed_query(chunk)
            embeddings.append(vector)
            self.document_map[self.current_id] = {
                'doc_id': document.id,
                'chunk': chunk,
                'title': document.title
            }
            self.current_id += 1
            
        embeddings_array = np.array(embeddings, dtype='float32')
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
            
        self.index.add(embeddings_array)
    
    def search(self, query: str, k: int = 3) -> List[dict]:
        query_vector = self.embeddings.embed_query(query)
        query_vector_array = np.array([query_vector], dtype='float32')
        
        if self.index is None:
            return []
            
        distances, indices = self.index.search(query_vector_array, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            doc_info = self.document_map[int(idx)]
            results.append({
                'doc_id': doc_info['doc_id'],
                'title': doc_info['title'],
                'chunk': doc_info['chunk'],
                'score': float(1 / (1 + distance))
            })
            
        return results