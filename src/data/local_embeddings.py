from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class LocalEmbeddings:
    """Local embeddings using Sentence-Transformers"""
    
    def __init__(self):
        logger.info("Initializing Sentence-Transformers model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
    
    def embed_query(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts at once"""
        return self.model.encode(texts).tolist() 