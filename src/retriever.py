"""
Retriever module for semantic search over complaint narratives.
Uses a FAISS vector index and SentenceTransformer embeddings to retrieve
the most relevant complaint text chunks for a given user query.
"""
from typing import List, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ComplaintRetriever:
    """
    Handles embedding of user queries and retrieval of relevant complaint chunks.
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: List[str],
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index = index
        self.metadata = metadata
        self.model = SentenceTransformer(model_name)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index size does not match metadata length."
            )

    def _search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper around FAISS search to satisfy static type checkers.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Distances and indices.
        """
        # FAISS's python binding has a C-level signature that some static
        # analysers (e.g. Pylance) misinterpret. At runtime `Index.search(x, k)`
        # returns `(distances, labels)`. Suppress the call-arg check here.
        return self.index.search(query_embedding, k)  # type: ignore[call-arg]

    def retrieve(self, question: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k most relevant complaint chunks.

        Parameters
        ----------
        question : str
            User query.
        k : int
            Number of chunks to retrieve.

        Returns
        -------
        List[str]
            Retrieved complaint text chunks.
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        query_embedding = self.model.encode([question])
        query_embedding = np.asarray(query_embedding, dtype="float32")

        _distances, indices = self._search(query_embedding, k)

        return [self.metadata[i] for i in indices[0]]
