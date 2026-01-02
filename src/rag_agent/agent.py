"""
Main RAG Agent for Intelligent Complaint Analysis
"""
from typing import List, Dict, Optional
import os
from .config import Config
from .data_loader import ComplaintLoader
from .vector_store import VectorStoreManager
from .query_engine import RAGQueryEngine


class ComplaintRAGAgent:
    """Main RAG Agent for analyzing customer complaints"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the RAG agent
        
        Args:
            data_path: Path to complaint data file
        """
        # Validate configuration
        Config.validate()
        
        self.data_path = data_path or Config.DATA_PATH
        self.vector_store_manager = VectorStoreManager()
        self.query_engine: Optional[RAGQueryEngine] = None
        self.products: List[str] = []
        self.is_initialized = False
    
    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the RAG agent by loading data and building vector store
        
        Args:
            force_rebuild: Force rebuild of vector store even if it exists
        """
        print("Initializing RAG Agent...")
        
        # Check if vector store already exists
        vector_store_path = Config.get_vector_store_path()
        vector_store_exists = self._check_vector_store_exists(vector_store_path)
        
        if vector_store_exists and not force_rebuild:
            print(f"Loading existing vector store from {vector_store_path}...")
            self.vector_store_manager.load_vector_store(vector_store_path)
            print("Vector store loaded successfully.")
        else:
            print("Building new vector store...")
            self._build_vector_store()
        
        # Initialize query engine
        self.query_engine = RAGQueryEngine(self.vector_store_manager)
        
        # Load product list
        self._load_products()
        
        self.is_initialized = True
        print(f"RAG Agent initialized successfully. Available products: {len(self.products)}")
    
    def _build_vector_store(self) -> None:
        """Build vector store from complaint data"""
        # Load and preprocess data
        print(f"Loading data from {self.data_path}...")
        loader = ComplaintLoader(self.data_path)
        loader.load_data()
        loader.preprocess()
        
        # Get documents
        documents = loader.get_documents()
        print(f"Loaded {len(documents)} complaints.")
        
        # Create vector store
        print("Creating embeddings and building vector store...")
        self.vector_store_manager.create_vector_store(documents)
        
        # Save vector store
        print("Saving vector store...")
        self.vector_store_manager.save_vector_store()
        print("Vector store saved successfully.")
        
        # Store products
        self.products = loader.get_products()
    
    def _check_vector_store_exists(self, path: str) -> bool:
        """Check if vector store exists at the given path"""
        if Config.VECTOR_STORE_TYPE == 'faiss':
            return os.path.exists(os.path.join(path, 'index.faiss'))
        elif Config.VECTOR_STORE_TYPE == 'chromadb':
            return os.path.exists(path) and os.path.isdir(path)
        return False
    
    def _load_products(self) -> None:
        """Load list of available products from data"""
        if not self.products:
            # Try to load from data file
            try:
                loader = ComplaintLoader(self.data_path)
                loader.load_data()
                self.products = loader.get_products()
            except Exception as e:
                print(f"Warning: Could not load products: {e}")
                self.products = []
    
    def ask(
        self,
        question: str,
        product: Optional[str] = None,
        k: int = None
    ) -> Dict[str, any]:
        """
        Ask a question about customer complaints
        
        Args:
            question: Plain-English question about complaints
            product: Optional product filter
            k: Number of relevant complaints to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("RAG Agent not initialized. Call initialize() first.")
        
        print(f"\nProcessing query: '{question}'")
        if product:
            print(f"Filtering by product: {product}")
        
        result = self.query_engine.query(
            question=question,
            product_filter=product,
            k=k
        )
        
        return result
    
    def compare(
        self,
        question: str,
        products: List[str],
        k_per_product: int = 3
    ) -> Dict[str, any]:
        """
        Compare complaints across multiple products
        
        Args:
            question: Comparison question
            products: List of products to compare
            k_per_product: Number of complaints per product
            
        Returns:
            Dictionary with comparative analysis
        """
        if not self.is_initialized:
            raise RuntimeError("RAG Agent not initialized. Call initialize() first.")
        
        print(f"\nComparing products: {', '.join(products)}")
        print(f"Question: '{question}'")
        
        result = self.query_engine.compare_products(
            question=question,
            products=products,
            k_per_product=k_per_product
        )
        
        return result
    
    def get_available_products(self) -> List[str]:
        """
        Get list of available products in the dataset
        
        Returns:
            List of product names
        """
        return self.products
    
    def rebuild_index(self) -> None:
        """Rebuild the vector store index from scratch"""
        print("Rebuilding vector store index...")
        self.is_initialized = False
        self.initialize(force_rebuild=True)
