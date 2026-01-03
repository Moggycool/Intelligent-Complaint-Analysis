"""Sentence-aware text chunker using NLTK."""
from typing import List, Dict, Optional, Any
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class TextChunker:
    """
    Sentence-aware text chunker using NLTK.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 1):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """ Chunk text into smaller pieces while preserving sentence boundaries."""
        if not text.strip():
            return []

        metadata = metadata if metadata is not None else {}
        sentences = nltk.sent_tokenize(text)

        chunks: List[Dict[str, Any]] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": metadata
                    })

                current_chunk = current_chunk[-self.overlap:] if self.overlap > 0 else []
                current_length = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk).strip(),
                "metadata": metadata
            })

        return chunks
