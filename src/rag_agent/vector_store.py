"""
Vector store management for semantic search
"""
from typing import List, Dict, Optional, Any
import pickle
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from .config import Config


class VectorStoreManager:
    """Manage vector store for semantic search"""
    
    def __init__(self, store_type: str = None):
        """
        Initialize vector store manager
        
        Args:
            store_type: Type of vector store ('faiss' or 'chromadb')
        """
        self.store_type = store_type or Config.VECTOR_STORE_TYPE
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.vector_store: Optional[Any] = None
    
    def create_vector_store(self, documents: List[Dict[str, str]]) -> Any:
        """
        Create vector store from documents
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            
        Returns:
            Vector store instance
        """
        # Convert to LangChain Document format
        langchain_docs = [
            Document(
                page_content=doc['text'],
                metadata=doc['metadata']
            )
            for doc in documents
        ]
        
        if self.store_type == 'faiss':
            self.vector_store = FAISS.from_documents(
                langchain_docs,
                self.embeddings
            )
        elif self.store_type == 'chromadb':
            self.vector_store = Chroma.from_documents(
                langchain_docs,
                self.embeddings,
                persist_directory=Config.get_vector_store_path()
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        return self.vector_store
    
    def save_vector_store(self, path: str = None) -> None:
        """
        Save vector store to disk
        
        Args:
            path: Path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        save_path = path or Config.get_vector_store_path()
        
        if self.store_type == 'faiss':
            # Save FAISS index
            self.vector_store.save_local(save_path)
        elif self.store_type == 'chromadb':
            # ChromaDB persists automatically
            pass
    
    def load_vector_store(self, path: str = None) -> Any:
        """
        Load vector store from disk
        
        Args:
            path: Path to load the vector store from
            
        Returns:
            Loaded vector store instance
        """
        load_path = path or Config.get_vector_store_path()
        
        if self.store_type == 'faiss':
            self.vector_store = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif self.store_type == 'chromadb':
            self.vector_store = Chroma(
                persist_directory=load_path,
                embedding_function=self.embeddings
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        return self.vector_store
    
    def search(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Perform semantic search
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {'product': 'Credit Card'})
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Load or create one first.")
        
        k = k or Config.TOP_K_RESULTS
        
        if filter_dict:
            results = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[tuple]:
        """
        Perform semantic search with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        k = k or Config.TOP_K_RESULTS
        
        if filter_dict:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return results
