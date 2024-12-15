"""
config.py
Configuration module for DocumentMentor
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

# Application Settings
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))

# Database Settings
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/documentmentor.db')

# Vector Store Settings
VECTOR_STORE_PATH = Path(os.getenv('VECTOR_STORE_PATH', 'data/vector_store'))

# Debug Mode
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Ensure necessary directories exist
def create_directories():
    """Create necessary directories if they don't exist"""
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)

# Create directories on module import
create_directories()