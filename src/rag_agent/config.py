"""
Configuration management for RAG agent
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration settings for the RAG agent"""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "vector_store/")
    
    # LLM Settings
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))
    
    # Retrieval Settings
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    
    # Data Settings
    DATA_PATH: str = os.getenv("DATA_PATH", "data/complaints.csv")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in .env file")
        return True
    
    @classmethod
    def get_vector_store_path(cls, create: bool = True) -> str:
        """Get vector store path and optionally create directory"""
        if create:
            os.makedirs(cls.VECTOR_STORE_PATH, exist_ok=True)
        return cls.VECTOR_STORE_PATH
