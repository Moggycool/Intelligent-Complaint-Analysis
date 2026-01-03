"""Sentence-transformer embedding model with NLTK-based text chunking."""
from typing import List
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

nltk.download("punkt", quiet=True)


class EmbeddingModel:
    """
    Sentence-transformer embedding model with NLTK-based text chunking.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_words: int = 150,
    ):
        self.model = SentenceTransformer(model_name)
        self.max_words = max_words

    def _chunk_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        chunks, current_chunk, length = [], [], 0

        for sentence in sentences:
            words = sentence.split()
            if length + len(words) <= self.max_words:
                current_chunk.append(sentence)
                length += len(words)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                length = len(words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Chunk texts using NLTK, then generate embeddings.
        """
        all_chunks = []
        for text in texts:
            all_chunks.extend(self._chunk_text(text))

        return self.model.encode(
            all_chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
