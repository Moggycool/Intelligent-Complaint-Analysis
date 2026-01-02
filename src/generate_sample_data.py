"""
Generate sample CFPB-like complaint data for testing purposes.

This creates a small sample dataset that mimics the structure of the CFPB complaint database.
Use this if you want to test the pipeline without downloading the full dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path


def generate_sample_data(n_samples=1000, output_path='data/complaints.csv'):
    """
    Generate sample CFPB-like complaint data
    
    Args:
        n_samples: Number of sample complaints to generate
        output_path: Path to save the sample dataset
    """
    print(f"Generating {n_samples} sample complaints...")
    
    # Define categories
    products = [
        'Credit card or prepaid card',
        'Checking or savings account',
        'Payday loan, title loan, or personal loan',
        'Money transfer, virtual currency, or money service',
        'Mortgage',
        'Student loan',
        'Debt collection',
        'Credit reporting'
    ]
    
    sub_products = {
        'Credit card or prepaid card': ['General-purpose credit card', 'Store credit card', 'Prepaid card'],
        'Checking or savings account': ['Checking account', 'Savings account', 'Other banking product'],
        'Payday loan, title loan, or personal loan': ['Personal loan', 'Payday loan', 'Title loan'],
        'Money transfer, virtual currency, or money service': ['Domestic (US) money transfer', 'International money transfer', 'Virtual currency'],
        'Mortgage': ['Conventional home mortgage', 'FHA mortgage', 'VA mortgage'],
        'Student loan': ['Federal student loan', 'Private student loan'],
        'Debt collection': ['Credit card debt', 'Medical debt', 'Other debt'],
        'Credit reporting': ['Credit reporting', 'Other personal consumer report']
    }
    
    issues = [
        'Unauthorized charges or transactions',
        'Problem with a purchase shown on statement',
        'Incorrect information on credit report',
        'Problem with customer service',
        'Account opening, closing, or management',
        'Managing an account',
        'Fees or interest',
        'Problem with a credit reporting company',
        'Improper use of credit report',
        'Attempts to collect debt not owed'
    ]
    
    companies = [
        'CAPITAL ONE FINANCIAL CORPORATION',
        'JPMORGAN CHASE & CO.',
        'CITIBANK, N.A.',
        'BANK OF AMERICA',
        'WELLS FARGO & COMPANY',
        'SYNCHRONY FINANCIAL',
        'DISCOVER BANK',
        'AMERICAN EXPRESS COMPANY',
        'PNC BANK N.A.',
        'U.S. BANCORP'
    ]
    
    states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA']
    
    # Sample complaint narratives
    narrative_templates = [
        "On {date}, I noticed unauthorized charges on my account totaling ${amount}. I contacted the company but received no response. This is unacceptable and needs to be resolved immediately.",
        "I have been a customer for {years} years and recently experienced issues with {issue}. Despite multiple calls to customer service, the problem remains unresolved. I am very disappointed with the service.",
        "My credit card statement shows a charge of ${amount} that I did not authorize. I reported this to the company on {date} but they have not taken any action to remove the charge.",
        "I opened an account with {company} expecting good service, but I have been charged excessive fees without proper notification. The fees total ${amount} over the past {months} months.",
        "There is incorrect information on my credit report that is damaging my credit score. I have disputed this multiple times with {company} but they refuse to correct it.",
        "I received collection calls for a debt I do not owe. The amount claimed is ${amount}. I have requested validation of this debt but received no response.",
        "My account was closed without proper notice or explanation. I had a balance of ${amount} and need clarification on how to access my funds.",
        "I applied for a loan and was denied based on incorrect credit information. The company pulled my credit report showing erroneous data that I had previously disputed.",
        "I made a payment of ${amount} on {date} but it was not properly credited to my account. This resulted in late fees and negative credit reporting.",
        "Customer service representatives have been unhelpful and rude when I tried to resolve issues with my account. I have called {times} times with no resolution."
    ]
    
    # Generate data
    data = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(n_samples):
        product = random.choice(products)
        sub_product = random.choice(sub_products.get(product, ['']))
        issue = random.choice(issues)
        company = random.choice(companies)
        state = random.choice(states)
        
        # Generate dates
        days_offset = random.randint(0, 1460)  # Up to 4 years
        date_received = base_date + timedelta(days=days_offset)
        date_sent = date_received + timedelta(days=random.randint(0, 15))
        
        # Generate narrative (80% chance of having one)
        if random.random() < 0.8:
            template = random.choice(narrative_templates)
            narrative = template.format(
                date=date_received.strftime('%m/%d/%Y'),
                amount=random.randint(10, 5000),
                years=random.randint(1, 20),
                months=random.randint(1, 24),
                issue=issue.lower(),
                company=company,
                times=random.randint(2, 10)
            )
        else:
            narrative = np.nan
        
        # Create complaint record
        complaint = {
            'Complaint ID': 100000 + i,
            'Product': product,
            'Sub-product': sub_product,
            'Issue': issue,
            'Sub-issue': random.choice(['', issue]),
            'Consumer complaint narrative': narrative,
            'Company': company,
            'State': state,
            'ZIP code': f"{random.randint(10000, 99999)}",
            'Date received': date_received.strftime('%Y-%m-%d'),
            'Date sent to company': date_sent.strftime('%Y-%m-%d'),
            'Company response to consumer': random.choice([
                'Closed with explanation',
                'Closed with monetary relief',
                'Closed with non-monetary relief',
                'Closed'
            ]),
            'Timely response?': random.choice(['Yes', 'No']),
            'Consumer disputed?': random.choice(['Yes', 'No', ''])
        }
        
        data.append(complaint)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Sample dataset created successfully!")
    print(f"Saved to: {output_file}")
    print(f"\nDataset Info:")
    print(f"  Total complaints: {len(df):,}")
    print(f"  Complaints with narratives: {df['Consumer complaint narrative'].notna().sum():,}")
    print(f"  Products: {df['Product'].nunique()}")
    print(f"\nProduct distribution:")
    print(df['Product'].value_counts())
    
    return df


def main():
    """Main execution function"""
    print("="*70)
    print("Sample CFPB Complaint Data Generator")
    print("="*70)
    print("\nThis utility creates sample data for testing the pipeline.")
    print("For production use, download the actual CFPB dataset using:")
    print("  python src/download_data.py")
    print("\n" + "="*70 + "\n")
    
    # Generate sample data
    generate_sample_data(n_samples=5000)
    
    print("\n" + "="*70)
    print("Next steps:")
    print("1. Run the EDA notebook: jupyter notebook notebooks/01_eda_preprocessing.ipynb")
    print("   OR run the script: python src/eda_preprocessing.py")
    print("2. Then run: python src/create_vector_store.py")
    print("="*70)


if __name__ == "__main__":
    main()
