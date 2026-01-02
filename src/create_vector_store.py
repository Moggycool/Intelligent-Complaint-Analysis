"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing

This script performs:
1. Stratified sampling from the cleaned dataset
2. Text chunking using RecursiveCharacterTextSplitter
3. Embedding generation using sentence-transformers
4. Vector store creation and persistence using FAISS
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import List, Dict
from tqdm import tqdm

# LangChain for text chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sentence transformers for embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector store
import faiss


class ComplaintVectorStore:
    """
    A class to handle text chunking, embedding, and vector store creation
    for CFPB complaint data.
    """
    
    def __init__(
        self,
        embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the ComplaintVectorStore.
        
        Args:
            embedding_model_name: Name of the sentence-transformer model to use
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Storage for chunks and metadata
        self.chunks = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        
    def stratified_sample(
        self,
        df: pd.DataFrame,
        n_samples: int = 12000,
        stratify_column: str = 'Product',
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create a stratified sample from the dataset.
        
        Args:
            df: Input DataFrame
            n_samples: Total number of samples to draw (10000-15000)
            stratify_column: Column to use for stratification
            random_state: Random seed for reproducibility
            
        Returns:
            Stratified sample DataFrame
        """
        print(f"\nCreating stratified sample of {n_samples} complaints...")
        print(f"Stratifying by: {stratify_column}")
        
        # Get product distribution
        product_counts = df[stratify_column].value_counts()
        print("\nOriginal product distribution:")
        print(product_counts)
        
        # Calculate proportional sample sizes
        total_count = len(df)
        samples_per_product = {}
        
        for product, count in product_counts.items():
            proportion = count / total_count
            sample_size = int(n_samples * proportion)
            # Ensure at least 1 sample per product
            sample_size = max(1, sample_size)
            samples_per_product[product] = min(sample_size, count)
        
        # Adjust if total doesn't match n_samples exactly
        current_total = sum(samples_per_product.values())
        if current_total < n_samples:
            # Add remaining samples to largest category
            largest_product = product_counts.index[0]
            samples_per_product[largest_product] += (n_samples - current_total)
        
        print("\nTarget sample sizes per product:")
        for product, size in samples_per_product.items():
            print(f"  {product}: {size}")
        
        # Perform stratified sampling
        sampled_dfs = []
        for product, sample_size in samples_per_product.items():
            product_df = df[df[stratify_column] == product]
            if len(product_df) >= sample_size:
                sampled = product_df.sample(n=sample_size, random_state=random_state)
            else:
                sampled = product_df
            sampled_dfs.append(sampled)
        
        # Combine samples
        df_sample = pd.concat(sampled_dfs, ignore_index=True)
        
        print(f"\nFinal sample size: {len(df_sample)}")
        print("\nSampled product distribution:")
        print(df_sample[stratify_column].value_counts())
        
        return df_sample
    
    def chunk_texts(self, df: pd.DataFrame, text_column: str = 'cleaned_narrative') -> None:
        """
        Split texts into chunks and store with metadata.
        
        Args:
            df: DataFrame containing texts
            text_column: Name of column containing text to chunk
        """
        print(f"\nChunking texts with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}...")
        
        self.chunks = []
        self.metadata = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
            text = row[text_column]
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            # Store each chunk with its metadata
            for chunk_idx, chunk in enumerate(text_chunks):
                self.chunks.append(chunk)
                self.metadata.append({
                    'complaint_id': row.get('Complaint ID', idx),
                    'product_category': row.get('Product', 'Unknown'),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(text_chunks),
                    'original_index': idx
                })
        
        print(f"Created {len(self.chunks)} chunks from {len(df)} complaints")
        print(f"Average chunks per complaint: {len(self.chunks) / len(df):.2f}")
    
    def generate_embeddings(self, batch_size: int = 32) -> None:
        """
        Generate embeddings for all chunks.
        
        Args:
            batch_size: Batch size for embedding generation
        """
        print(f"\nGenerating embeddings using {self.embedding_model_name}...")
        
        # Generate embeddings in batches
        embeddings_list = []
        
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Embedding"):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings_list.append(batch_embeddings)
        
        # Combine all embeddings
        self.embeddings = np.vstack(embeddings_list)
        
        print(f"Generated embeddings shape: {self.embeddings.shape}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
    
    def create_vector_store(self) -> None:
        """
        Create FAISS index from embeddings.
        """
        print("\nCreating FAISS vector store...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index (using IndexFlatIP for inner product/cosine similarity)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"FAISS index created with {self.index.ntotal} vectors")
    
    def save_vector_store(self, output_dir: str = '../vector_store') -> None:
        """
        Save the vector store and metadata to disk.
        
        Args:
            output_dir: Directory to save vector store files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving vector store to {output_path}...")
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_path / 'complaint_vectors.index'))
        
        # Save metadata and chunks
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.metadata,
                'embedding_model': self.embedding_model_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }, f)
        
        # Save configuration
        config = {
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embeddings.shape[1],
            'index_type': 'FAISS IndexFlatIP'
        }
        
        with open(output_path / 'config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print("Vector store saved successfully!")
        print(f"Files saved:")
        print(f"  - complaint_vectors.index (FAISS index)")
        print(f"  - metadata.pkl (chunks and metadata)")
        print(f"  - config.pkl (configuration)")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks given a query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'chunk': self.chunks[idx],
                'metadata': self.metadata[idx],
                'score': float(distances[0][i])
            })
        
        return results


def main():
    """
    Main execution function for Task 2.
    """
    print("="*70)
    print("Task 2: Text Chunking, Embedding, and Vector Store Indexing")
    print("="*70)
    
    # Load cleaned dataset
    data_path = Path('data/filtered_complaints.csv')
    
    if not data_path.exists():
        # Try relative path if running from src/
        data_path = Path('../data/filtered_complaints.csv')
    
    if not data_path.exists():
        print(f"Error: filtered_complaints.csv not found!")
        print("Please run Task 1 (EDA and preprocessing) first.")
        return
    
    print(f"\nLoading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} complaints")
    
    # Initialize vector store
    vector_store = ComplaintVectorStore(
        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
        chunk_size=512,  # Balanced size for semantic coherence
        chunk_overlap=50  # Small overlap to maintain context
    )
    
    # Create stratified sample
    df_sample = vector_store.stratified_sample(
        df,
        n_samples=12000,  # Middle of 10k-15k range
        stratify_column='Product',
        random_state=42
    )
    
    # Chunk texts
    vector_store.chunk_texts(df_sample, text_column='cleaned_narrative')
    
    # Generate embeddings
    vector_store.generate_embeddings(batch_size=32)
    
    # Create vector store
    vector_store.create_vector_store()
    
    # Save vector store
    output_dir = 'vector_store' if Path('vector_store').exists() else '../vector_store'
    vector_store.save_vector_store(output_dir=output_dir)
    
    # Test search functionality
    print("\n" + "="*70)
    print("Testing search functionality...")
    print("="*70)
    
    test_query = "unauthorized charges on my credit card"
    print(f"\nTest query: '{test_query}'")
    results = vector_store.search(test_query, k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (score: {result['score']:.4f}):")
        print(f"Product: {result['metadata']['product_category']}")
        print(f"Complaint ID: {result['metadata']['complaint_id']}")
        print(f"Chunk: {result['chunk'][:200]}...")
    
    print("\n" + "="*70)
    print("Task 2 completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
