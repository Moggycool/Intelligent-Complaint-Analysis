"""LLM-based answer generation module for the RAG pipeline."""
from transformers import pipeline


class AnswerGenerator:
    """
    Generates grounded answers using a large language model.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        max_new_tokens: int = 300,
        temperature: float = 0.2
    ):
        """
        Parameters
        ----------
        model_name : str
            Hugging Face model identifier.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature (lower = more factual).
        """
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    def generate(self, prompt: str) -> str:
        """
        Generate an answer from the LLM.

        Parameters
        ----------
        prompt : str
            Fully constructed RAG prompt.

        Returns
        -------
        str
            Generated answer text.
        """
        output = self.generator(prompt)[0]["generated_text"]
        return output.split("Answer:")[-1].strip()
