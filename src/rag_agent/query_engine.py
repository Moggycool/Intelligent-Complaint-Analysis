"""
RAG query engine with LLM integration
"""
from typing import List, Dict, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from .config import Config
from .vector_store import VectorStoreManager


class RAGQueryEngine:
    """RAG-based query engine for complaint analysis"""
    
    # Prompt template for complaint analysis
    ANALYSIS_PROMPT_TEMPLATE = """You are an expert analyst helping internal users understand customer complaints about financial services.

Based on the following customer complaint narratives, answer the user's question with clear, actionable insights.

User Question: {question}

Relevant Complaint Narratives:
{context}

Provide a concise, insightful answer that:
1. Directly addresses the user's question
2. Identifies common themes and patterns in the complaints
3. Highlights the main issues customers are facing
4. Uses specific examples from the narratives when relevant
5. Keeps the response under 300 words

Answer:"""
    
    COMPARISON_PROMPT_TEMPLATE = """You are an expert analyst comparing customer complaints across different financial products.

User Question: {question}

Product-specific Complaint Narratives:
{context}

Provide a comparative analysis that:
1. Highlights key differences in complaint patterns across products
2. Identifies product-specific issues
3. Notes any common themes across products
4. Uses specific examples from the narratives
5. Keeps the response under 300 words

Answer:"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initialize RAG query engine
        
        Args:
            vector_store_manager: Initialized vector store manager
        """
        self.vector_store_manager = vector_store_manager
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def query(
        self,
        question: str,
        product_filter: Optional[str] = None,
        k: int = None
    ) -> Dict[str, any]:
        """
        Process a user query and generate an answer
        
        Args:
            question: User's question about complaints
            product_filter: Optional product name to filter results
            k: Number of relevant complaints to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        # Prepare filter
        filter_dict = None
        if product_filter:
            filter_dict = {'product': product_filter}
        
        # Retrieve relevant documents
        relevant_docs = self.vector_store_manager.search(
            query=question,
            k=k or Config.TOP_K_RESULTS,
            filter_dict=filter_dict
        )
        
        # Format context from retrieved documents
        context = self._format_context(relevant_docs)
        
        # Generate answer using LLM
        prompt = PromptTemplate(
            template=self.ANALYSIS_PROMPT_TEMPLATE,
            input_variables=["question", "context"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(question=question, context=context)
        
        return {
            'answer': answer.strip(),
            'sources': relevant_docs,
            'product_filter': product_filter,
            'num_sources': len(relevant_docs)
        }
    
    def compare_products(
        self,
        question: str,
        products: List[str],
        k_per_product: int = 3
    ) -> Dict[str, any]:
        """
        Compare complaints across multiple products
        
        Args:
            question: Comparison question
            products: List of product names to compare
            k_per_product: Number of complaints to retrieve per product
            
        Returns:
            Dictionary with comparative answer and sources
        """
        all_docs = []
        product_contexts = []
        
        # Retrieve documents for each product
        for product in products:
            docs = self.vector_store_manager.search(
                query=question,
                k=k_per_product,
                filter_dict={'product': product}
            )
            all_docs.extend(docs)
            
            # Format product-specific context
            product_context = f"\n=== {product} Complaints ===\n"
            product_context += self._format_context(docs)
            product_contexts.append(product_context)
        
        # Combine all contexts
        combined_context = "\n".join(product_contexts)
        
        # Generate comparative answer
        prompt = PromptTemplate(
            template=self.COMPARISON_PROMPT_TEMPLATE,
            input_variables=["question", "context"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(question=question, context=combined_context)
        
        return {
            'answer': answer.strip(),
            'sources': all_docs,
            'products_compared': products,
            'num_sources': len(all_docs)
        }
    
    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant complaints found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            product = doc.metadata.get('product', 'Unknown')
            issue = doc.metadata.get('issue', 'Unknown')
            
            context_part = f"\nComplaint {i} (Product: {product}, Issue: {issue}):\n{doc.page_content[:500]}..."
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
