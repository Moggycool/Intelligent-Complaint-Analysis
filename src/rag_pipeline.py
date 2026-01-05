"""End-to-end Retrieval-Augmented Generation pipeline."""
from typing import Dict
from src.prompts import build_prompt


class RAGPipeline:
    """
    Orchestrates retrieval, prompt construction, and answer generation.
    """

    def __init__(self, retriever, generator):
        """
        Parameters
        ----------
        retriever : ComplaintRetriever
            Retriever instance.
        generator : AnswerGenerator
            Generator instance.
        """
        self.retriever = retriever
        self.generator = generator

    def run(self, question: str, k: int = 5) -> Dict:
        """
        Execute the full RAG pipeline.

        Parameters
        ----------
        question : str
            User query.
        k : int
            Number of retrieved chunks.

        Returns
        -------
        Dict
            Dictionary containing question, answer, and sources.
        """
        retrieved_chunks = self.retriever.retrieve(question, k)
        prompt = build_prompt(retrieved_chunks, question)
        answer = self.generator.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks[:2]
        }
