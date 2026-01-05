"""Embedding model with text chunking and metadata handling."""
from typing import List, Dict, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download("punkt", quiet=True)


class EmbeddingModel:
    """
    Sentence-transformer embedding model with automatic text chunking
    and metadata handling for each chunk.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 5, overlap: int = 1):
        """
        Args:
            model_name (str): SentenceTransformer model name
            chunk_size (int): Number of sentences per chunk
            overlap (int): Number of overlapping sentences between chunks
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits a text into overlapping chunks based on sentences.

        Args:
            text (str): The text to chunk

        Returns:
            List[str]: List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(sentences), step):
            chunk = " ".join(sentences[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def embed_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Returns embeddings for text chunks with optional metadata attached.

        Args:
            text (str): The text to embed
            metadata (Dict, optional): Additional metadata for each chunk

        Returns:
            List[Dict]: Each dict contains 'chunk', 'embedding', and 'metadata'
        """
        chunks = self.chunk_text(text)
        embeddings = self.model.encode(chunks)
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk": chunk,
                "embedding": embeddings[i],
                "metadata": metadata or {}
            }
            result.append(chunk_data)
        return result
