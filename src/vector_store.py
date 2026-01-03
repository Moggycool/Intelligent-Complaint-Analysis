"""FAISS vector store with metadata persistence."""
import os
from typing import List, Dict, Any
import pickle
import faiss
import numpy as np


class FaissVectorStore:
    """
    FAISS vector store with metadata persistence.
    Supports adding embeddings, saving/loading, and searching.
    """

    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and their metadata to the store.

        Args:
            embeddings: np.ndarray of shape (n_samples, embedding_dim)
            metadatas: List of dictionaries with metadata for each embedding
        """
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")

        # FAISS requires float32 and contiguous array
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Pylance often complains, suppress false positive
        self.index.add(embeddings)  # type: ignore[arg-type]
        self.metadata.extend(metadatas)

    def save(self, path: str) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            path: Directory path to save index and metadata
        """
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str) -> "FaissVectorStore":
        """
        Load a FAISS index and metadata from disk.

        Args:
            path: Directory path containing index.faiss and metadata.pkl

        Returns:
            FaissVectorStore instance
        """
        index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.metadata = metadata
        return store

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for the top-k nearest neighbors.

        Args:
            query_embeddings: np.ndarray of shape (1, embedding_dim) or (n_queries, embedding_dim)
            k: Number of nearest neighbors to retrieve

        Returns:
            List of dictionaries with 'score' and 'metadata'
        """
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        scores, indices = self.index.search(
            query_embeddings, k)  # type: ignore[arg-type]

        results = []
        for query_idx in range(indices.shape[0]):
            query_results = []
            for idx, score in zip(indices[query_idx], scores[query_idx]):
                if idx == -1:
                    continue
                query_results.append({
                    "score": float(score),
                    "metadata": self.metadata[idx]
                })
            results.append(query_results)
        return results
