""" Retriever module for semantic search over customer complaint excerpts. """
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class Retriever:
    """
    Minimal production Retriever wired for:
      - index: either a FAISS-like index with .search(X, k) -> (D, I),
               OR None (then we do brute-force over embeddings)
      - metadata: list of dicts aligned with rows in embeddings / index ids
      - model_name: sentence-transformers model to encode query
      - embeddings: optional precomputed embeddings (N x dim) as np.ndarray

    Returns List[Dict[str, Any]] with at least:
      - text: str
      - score: float  (note: meaning depends on backend; see `score_type`)
      - plus any metadata fields (complaint_id, product, doc_id, etc.)
    """

    def __init__(
        self,
        index: Any = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings: Optional[np.ndarray] = None,
        normalize: bool = True,
        # If True, for FAISS L2 distances we convert to a similarity-like score in (0,1]
        # via 1/(1+dist). This makes scores easier to interpret and more consistent.
        convert_faiss_distance_to_similarity: bool = True,
    ) -> None:
        self.index = index
        self.metadata = metadata or []
        self.model_name = model_name
        self.embeddings = embeddings
        self.normalize = normalize
        self.convert_faiss_distance_to_similarity = convert_faiss_distance_to_similarity

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for Retriever. "
                "Install: pip install sentence-transformers"
            )
        self.encoder = SentenceTransformer(model_name)

        # Basic validation (helps catch silent miswiring)
        if self.index is None and self.embeddings is None:
            raise ValueError(
                "Provide either `index` (FAISS-like) or `embeddings` (np.ndarray).")

        if self.embeddings is not None and len(self.metadata) not in (0, self.embeddings.shape[0]):
            raise ValueError(
                f"metadata length ({len(self.metadata)}) must match embeddings rows ({self.embeddings.shape[0]}), "
                "or metadata can be empty."
            )

    def _encode_query(self, query: str) -> np.ndarray:
        q_emb = self.encoder.encode(
            [query], convert_to_numpy=True).astype("float32")
        if self.normalize:
            q_emb = q_emb / \
                (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        return q_emb

    @staticmethod
    def _extract_text(meta: Dict[str, Any]) -> str:
        return (meta.get("text") or meta.get("chunk_text") or meta.get("content") or "").strip()

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not query or k <= 0:
            return []

        q_emb = self._encode_query(query)

        # Path 1: FAISS-like index
        if self.index is not None and hasattr(self.index, "search"):
            # Some FAISS variants require k <= ntotal
            try:
                ntotal = int(getattr(self.index, "ntotal"))
                k_eff = min(k, max(ntotal, 0)) if ntotal > 0 else k
                if k_eff <= 0:
                    return []
            except Exception:
                k_eff = k

            D, I = self.index.search(q_emb, k_eff)  # shapes: (1,k)
            hits: List[Dict[str, Any]] = []

            for rank, (idx, dist) in enumerate(zip(I[0].tolist(), D[0].tolist())):
                if idx is None or idx < 0:
                    continue

                meta = self.metadata[idx] if idx < len(self.metadata) else {}
                text = self._extract_text(meta)

                # `dist` meaning depends on index metric. Common cases:
                # - IndexFlatL2: smaller distance is better
                # - IndexFlatIP / cosine on normalized vectors: larger is better
                score_type = "faiss_distance_or_similarity"

                score_val: float
                if self.convert_faiss_distance_to_similarity:
                    # Heuristic: if distance is non-negative, treat as distance and convert.
                    # If it’s negative (can happen with inner product), keep raw.
                    if isinstance(dist, (int, float)) and dist >= 0:
                        # (0,1], higher is better
                        score_val = float(1.0 / (1.0 + dist))
                        score_type = "faiss_distance_converted_to_similarity"
                    else:
                        score_val = float(dist)
                else:
                    score_val = float(dist)

                hits.append(
                    {
                        **meta,
                        "text": text,
                        "score": score_val,
                        "score_type": score_type,
                        "rank": rank,
                        "index_id": int(idx),
                    }
                )
            return hits

        # Path 2: brute-force over embeddings
        emb = self.embeddings
        if emb is None:
            return []

        if emb.ndim != 2:
            raise ValueError(
                f"embeddings must be 2D array (N, dim); got shape={emb.shape}")

        # Cosine similarity if normalized; otherwise dot product.
        if self.normalize:
            emb_norm = emb / \
                (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            sims = (emb_norm @ q_emb[0]).astype("float32")  # (N,)
            score_type = "cosine_similarity"
        else:
            sims = (emb @ q_emb[0]).astype("float32")
            score_type = "dot_product"

        k_eff = min(k, sims.shape[0])
        top_idx = np.argsort(-sims)[:k_eff]

        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_idx.tolist()):
            meta = self.metadata[idx] if idx < len(self.metadata) else {}
            text = self._extract_text(meta)
            hits.append(
                {
                    **meta,
                    "text": text,
                    "score": float(sims[idx]),
                    "score_type": score_type,
                    "rank": rank,
                    "index_id": int(idx),
                }
            )
        return hits


def dynamic_k(question: str, base_k: int = 5) -> int:
    q = (question or "").lower()

    aggregation_markers = [
        "most common",
        "common issues",
        "recurring",
        "repeated",
        "frequent",
        "typically",
        "in general",
        "overall",
        "trend",
        "trends",
        "themes",
        "patterns",
        "what are customers complaining",
        "what do customers complain",
        "what issues do customers report",
        "what issues are customers reporting",
        "summarize",
        "summary of",
        "top issues",
        "main issues",
    ]

    # For broad/aggregate questions, pull more evidence.
    if any(m in q for m in aggregation_markers):
        return max(base_k, 20)

    return base_k
