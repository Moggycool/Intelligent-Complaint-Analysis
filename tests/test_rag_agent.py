"""
Unit tests for the RAG Agent
"""
import unittest
import os
import sys
import json
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_agent.data_loader import ComplaintLoader
from rag_agent.config import Config


class TestDataLoader(unittest.TestCase):
    """Test cases for ComplaintLoader"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary test data
        self.test_data = [
            {
                "complaint_narrative": "Test complaint about credit card fees",
                "product": "Credit Card",
                "company": "Test Bank",
                "state": "CA",
                "issue": "Fees"
            },
            {
                "complaint_narrative": "Another test complaint",
                "product": "Mortgage",
                "company": "Test Lender",
                "state": "NY",
                "issue": "Delays"
            }
        ]
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()
        
        self.loader = ComplaintLoader(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_file.name)
    
    def test_load_data(self):
        """Test data loading"""
        data = self.loader.load_data()
        self.assertEqual(len(data), 2)
        self.assertIn('complaint_narrative', data.columns)
        self.assertIn('product', data.columns)
    
    def test_preprocess(self):
        """Test data preprocessing"""
        self.loader.load_data()
        data = self.loader.preprocess()
        
        # Check that data is cleaned
        self.assertEqual(len(data), 2)
        self.assertTrue(all(isinstance(text, str) for text in data['complaint_narrative']))
    
    def test_get_documents(self):
        """Test document extraction"""
        self.loader.load_data()
        self.loader.preprocess()
        docs = self.loader.get_documents()
        
        self.assertEqual(len(docs), 2)
        self.assertIn('text', docs[0])
        self.assertIn('metadata', docs[0])
        self.assertEqual(docs[0]['metadata']['product'], 'Credit Card')
    
    def test_get_products(self):
        """Test product extraction"""
        self.loader.load_data()
        products = self.loader.get_products()
        
        self.assertEqual(len(products), 2)
        self.assertIn('Credit Card', products)
        self.assertIn('Mortgage', products)
    
    def test_clean_text(self):
        """Test text cleaning"""
        dirty_text = "  Test   with   extra   spaces  "
        clean = self.loader._clean_text(dirty_text)
        self.assertEqual(clean, "Test with extra spaces")


class TestConfig(unittest.TestCase):
    """Test cases for Config"""
    
    def test_default_values(self):
        """Test default configuration values"""
        self.assertEqual(Config.VECTOR_STORE_TYPE, 'faiss')
        self.assertEqual(Config.LLM_MODEL, 'gpt-3.5-turbo')
        self.assertEqual(Config.TOP_K_RESULTS, 5)
    
    def test_get_vector_store_path(self):
        """Test vector store path creation"""
        # Create temp directory for test
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, 'test_store')
        
        try:
            # Save original config
            from unittest.mock import patch
            
            with patch.object(Config, 'VECTOR_STORE_PATH', test_path):
                path = Config.get_vector_store_path(create=True)
                
                self.assertTrue(os.path.exists(path))
                self.assertTrue(os.path.isdir(path))
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
