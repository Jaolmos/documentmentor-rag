import json
import faiss
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from src.utils.config import OPENAI_API_KEY, VECTOR_STORE_PATH
from src.core.document_processor import ProcessedDocument

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages document embeddings and semantic search using FAISS"""
    
    def __init__(self, embeddings: Optional[Embeddings] = None):
        self.embeddings = embeddings or OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.index = None
        self.document_map: Dict[int, str] = {}
        self.current_id = 0
        self.index_path = VECTOR_STORE_PATH / "faiss.index"
        self.document_map_path = VECTOR_STORE_PATH / "document_map.json"
        self.load_index()  # Try to load existing index
    
    def save_index(self):
        """Save FAISS index and document map to disk"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.document_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.document_map, f, ensure_ascii=False, indent=2)
            logger.info(f"Index and document map saved to {VECTOR_STORE_PATH}")
    
    def load_index(self):
        """Load FAISS index and document map if they exist"""
        try:
            if self.index_path.exists() and self.document_map_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                with open(self.document_map_path, 'r', encoding='utf-8') as f:
                    self.document_map = json.load(f)
                self.current_id = max(map(int, self.document_map.keys())) + 1 if self.document_map else 0
                logger.info("FAISS index and document map loaded successfully")
            else:
                logger.info("No previous index found. Will create a new one.")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.index = None
            self.document_map = {}
    
    def add_document(self, document: ProcessedDocument) -> None:
        """Add a processed document to the vector store"""
        logger.info(f"Processing document: {document.title}")
        logger.info(f"Number of chunks to process: {len(document.chunks)}")
        
        embeddings = []
        for i, chunk in enumerate(document.chunks, 1):
            if i % 5 == 0:  # Log every 5 chunks
                logger.info(f"Processing chunk {i}/{len(document.chunks)}")
            
            vector = self.embeddings.embed_query(chunk)
            embeddings.append(vector)
            self.document_map[self.current_id] = {
                'doc_id': document.id,
                'chunk': chunk,
                'title': document.title
            }
            self.current_id += 1
        
        logger.info("Creating/updating FAISS index...")
        embeddings_array = np.array(embeddings, dtype='float32')
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        
        self.index.add(embeddings_array)
        self.save_index()
        logger.info("Document processed and saved successfully")
    
    def search(self, query: str, k: int = 3) -> List[dict]:
        """Search for similar documents using semantic similarity"""
        if self.index is None:
            raise ValueError("No documents have been indexed")
        
        try:
            logger.info(f"Searching for: {query}")
            query_vector = self.embeddings.embed_query(query)
            query_vector_array = np.array([query_vector], dtype='float32')
            
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
            
            logger.info(f"Found {len(results)} relevant results")
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise

    async def asearch(self, query: str, k: int = 3) -> List[dict]:
        """Async version of search method"""
        return self.search(query, k)