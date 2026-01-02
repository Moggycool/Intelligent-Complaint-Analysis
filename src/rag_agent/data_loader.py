"""
Data loading and preprocessing for complaint narratives
"""
import pandas as pd
from typing import List, Dict, Optional
import json
import re


class ComplaintLoader:
    """Load and preprocess complaint data"""
    
    def __init__(self, file_path: str):
        """
        Initialize the complaint loader
        
        Args:
            file_path: Path to the complaints data file (CSV or JSON)
        """
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load complaint data from file
        
        Returns:
            DataFrame containing complaint data
        """
        if self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.json'):
            self.data = pd.read_json(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        return self.data
    
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess complaint data
        
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Remove rows with missing narratives
        if 'complaint_narrative' in self.data.columns:
            self.data = self.data.dropna(subset=['complaint_narrative'])
        elif 'narrative' in self.data.columns:
            self.data = self.data.rename(columns={'narrative': 'complaint_narrative'})
            self.data = self.data.dropna(subset=['complaint_narrative'])
        
        # Clean text
        self.data['complaint_narrative'] = self.data['complaint_narrative'].apply(
            self._clean_text
        )
        
        # Ensure product column exists
        if 'product' not in self.data.columns and 'Product' in self.data.columns:
            self.data = self.data.rename(columns={'Product': 'product'})
        
        return self.data
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean complaint narrative text
        
        Args:
            text: Raw complaint text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_documents(self) -> List[Dict[str, str]]:
        """
        Convert DataFrame to list of documents for vector store
        
        Returns:
            List of document dictionaries
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() and preprocess() first.")
        
        documents = []
        for idx, row in self.data.iterrows():
            doc = {
                'id': str(idx),
                'text': row.get('complaint_narrative', ''),
                'product': row.get('product', 'Unknown'),
                'metadata': {
                    'product': row.get('product', 'Unknown'),
                    'company': row.get('company', 'Unknown'),
                    'state': row.get('state', 'Unknown'),
                    'issue': row.get('issue', 'Unknown'),
                }
            }
            
            # Add any additional metadata fields
            for col in self.data.columns:
                if col not in ['complaint_narrative', 'product', 'company', 'state', 'issue']:
                    doc['metadata'][col] = row.get(col, '')
            
            documents.append(doc)
        
        return documents
    
    def get_products(self) -> List[str]:
        """
        Get list of unique products in the dataset
        
        Returns:
            List of product names
        """
        if self.data is None:
            raise ValueError("Data not loaded.")
        
        if 'product' in self.data.columns:
            return self.data['product'].unique().tolist()
        return []
