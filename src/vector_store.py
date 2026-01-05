"""FAISS vector store with automatic chunked embeddings and metadata persistence."""
import os
from pathlib import Path
from typing import List, Dict, Any
import pickle
import faiss
import numpy as np


class FaissVectorStore:
    """
    FAISS vector store with metadata persistence.
    Supports adding documents with chunked embeddings, saving/loading, and searching.
    """

    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict[str, Any]] = []

    def add_document_chunks(self, doc_chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks with embeddings and metadata.

        Args:
            doc_chunks: List of dicts from EmbeddingModel.embed_text()
                        Each dict must have keys: 'chunk', 'embedding', 'metadata'
        """
        embeddings = []
        metadatas = []

        for chunk_data in doc_chunks:
            embeddings.append(chunk_data["embedding"])
            # Merge chunk text into metadata
            meta = {**chunk_data.get("metadata", {}),
                    "chunk_text": chunk_data["chunk"]}
            metadatas.append(meta)

        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Add to FAISS index
        self.index.add(embeddings)  # type: ignore[arg-type]
        self.metadata.extend(metadatas)

    def save(self, path: str | Path) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            path: Directory path where index.faiss and metadata.pkl will be saved
        """
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "metadata.pkl", "wb") as f:
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

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Search the FAISS index for the top-k nearest neighbors.

        Args:
            query_embeddings: np.ndarray of shape (1, embedding_dim) or (n_queries, embedding_dim)
            k: Number of nearest neighbors to retrieve

        Returns:
            List[List[Dict]]: Each query returns a list of dicts with 'score' and 'metadata'
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
