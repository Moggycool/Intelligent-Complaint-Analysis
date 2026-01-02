#!/usr/bin/env python3
"""
Demo script to showcase the RAG Agent functionality
This demonstrates the data loading and retrieval capabilities
without requiring an OpenAI API key
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_agent.data_loader import ComplaintLoader

def main():
    print("=" * 80)
    print("Intelligent Complaint Analysis - RAG Agent Demo")
    print("=" * 80)
    
    # Load sample data
    print("\n1. Loading Sample Complaint Data...")
    loader = ComplaintLoader('data/sample_complaints.json')
    loader.load_data()
    loader.preprocess()
    
    docs = loader.get_documents()
    products = loader.get_products()
    
    print(f"✓ Successfully loaded {len(docs)} complaints")
    print(f"✓ Found {len(products)} unique products")
    
    # Display available products
    print("\n2. Available Products:")
    print("-" * 80)
    for product in sorted(products):
        count = sum(1 for doc in docs if doc['metadata']['product'] == product)
        print(f"   • {product}: {count} complaints")
    
    # Show sample complaints
    print("\n3. Sample Complaints:")
    print("-" * 80)
    
    for i, doc in enumerate(docs[:3], 1):
        print(f"\nComplaint #{i}:")
        print(f"   Product: {doc['metadata']['product']}")
        print(f"   Issue: {doc['metadata']['issue']}")
        print(f"   Company: {doc['metadata']['company']}")
        print(f"   State: {doc['metadata']['state']}")
        print(f"   Text: {doc['text'][:150]}...")
    
    # Show example queries
    print("\n4. Example Queries (require OpenAI API key):")
    print("-" * 80)
    print("   • python src/cli.py ask \"Why are people unhappy with Credit Cards?\"")
    print("   • python src/cli.py ask \"What are the main issues?\" --product \"Mortgage\"")
    print("   • python src/cli.py compare \"Compare complaint patterns\" \"Credit Card\" \"Mortgage\"")
    print("   • python src/cli.py interactive")
    
    # Data statistics
    print("\n5. Dataset Statistics:")
    print("-" * 80)
    
    # Count by product
    print("\n   Complaints by Product:")
    for product in sorted(products):
        count = sum(1 for doc in docs if doc['metadata']['product'] == product)
        percentage = (count / len(docs)) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {product:20} {count:2} {bar} {percentage:.1f}%")
    
    # Common issues
    print("\n   Top Issues:")
    issues = {}
    for doc in docs:
        issue = doc['metadata']['issue']
        issues[issue] = issues.get(issue, 0) + 1
    
    for issue, count in sorted(issues.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   • {issue}: {count} complaint(s)")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nTo use the full RAG Agent:")
    print("1. Set up your OpenAI API key in .env file")
    print("2. Run: python src/cli.py init --data-path data/sample_complaints.json")
    print("3. Start asking questions!")
    print()

if __name__ == '__main__':
    main()
