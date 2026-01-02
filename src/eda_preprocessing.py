"""
Task 1: Exploratory Data Analysis and Data Preprocessing (Script Version)

This script performs the same operations as the notebook but as a standalone Python script.
Run this if you prefer not to use Jupyter notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def count_words(text):
    """Count words in a text string"""
    if pd.isna(text):
        return 0
    return len(str(text).split())


def clean_text(text):
    """
    Clean complaint narrative text:
    - Convert to lowercase
    - Remove special characters
    - Remove boilerplate text
    - Normalize whitespace
    """
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove common boilerplate phrases
    boilerplate_phrases = [
        r'i am writing to file a complaint\s*',
        r'dear sir or madam\s*',
        r'to whom it may concern\s*',
        r'i am writing to\s*',
        r'xx+',  # Remove sequences of X's (often used for redaction)
    ]
    
    for phrase in boilerplate_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?\-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def load_data(data_path='data/complaints.csv'):
    """Load the CFPB complaint dataset"""
    print("="*70)
    print("Task 1: Exploratory Data Analysis and Data Preprocessing")
    print("="*70)
    
    data_file = Path(data_path)
    
    if not data_file.exists():
        print(f"\nError: Dataset not found at {data_path}")
        print("Please run src/download_data.py first or download manually from:")
        print("https://www.consumerfinance.gov/data-research/consumer-complaints/")
        return None
    
    print(f"\nLoading dataset from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    
    return df


def perform_eda(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    # Product distribution
    print("\n1. Product Distribution (Top 20):")
    print("-" * 70)
    product_counts = df['Product'].value_counts()
    print(product_counts.head(20))
    
    # Narrative analysis
    print("\n2. Narrative Analysis:")
    print("-" * 70)
    total_complaints = len(df)
    complaints_with_narrative = df['Consumer complaint narrative'].notna().sum()
    complaints_without_narrative = df['Consumer complaint narrative'].isna().sum()
    
    print(f"Total Complaints: {total_complaints:,}")
    print(f"Complaints WITH narratives: {complaints_with_narrative:,} ({complaints_with_narrative/total_complaints*100:.2f}%)")
    print(f"Complaints WITHOUT narratives: {complaints_without_narrative:,} ({complaints_without_narrative/total_complaints*100:.2f}%)")
    
    # Word count analysis
    print("\n3. Calculating narrative word counts...")
    df['narrative_word_count'] = df['Consumer complaint narrative'].apply(count_words)
    df_with_narrative = df[df['narrative_word_count'] > 0].copy()
    
    print("\nNarrative Word Count Statistics:")
    print(df_with_narrative['narrative_word_count'].describe())
    
    # Identify very short and long narratives
    very_short_threshold = 10
    very_long_threshold = df_with_narrative['narrative_word_count'].quantile(0.95)
    
    very_short = df_with_narrative[df_with_narrative['narrative_word_count'] < very_short_threshold]
    very_long = df_with_narrative[df_with_narrative['narrative_word_count'] > very_long_threshold]
    
    print(f"\nVery short narratives (< {very_short_threshold} words): {len(very_short):,} ({len(very_short)/len(df_with_narrative)*100:.2f}%)")
    print(f"Very long narratives (> {very_long_threshold:.0f} words): {len(very_long):,} ({len(very_long)/len(df_with_narrative)*100:.2f}%)")
    
    return df


def filter_data(df):
    """Filter dataset to target products and remove empty narratives"""
    print("\n" + "="*70)
    print("DATA FILTERING")
    print("="*70)
    
    print(f"\nOriginal dataset size: {len(df):,}")
    
    # Identify matching products
    all_products = df['Product'].unique()
    matching_products = [p for p in all_products if any(keyword in p.lower() for keyword in 
                         ['credit card', 'personal loan', 'payday loan', 'savings', 'checking', 'money transfer'])]
    
    print("\nMatching products found:")
    for product in matching_products:
        count = len(df[df['Product'] == product])
        print(f"  {product}: {count:,} complaints")
    
    # Filter by products
    df_filtered = df[df['Product'].isin(matching_products)].copy()
    print(f"\nAfter product filtering: {len(df_filtered):,}")
    
    # Remove empty narratives
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notna()].copy()
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].astype(str).str.strip() != ''].copy()
    print(f"After removing empty narratives: {len(df_filtered):,}")
    
    return df_filtered


def clean_data(df):
    """Clean text narratives"""
    print("\n" + "="*70)
    print("TEXT CLEANING")
    print("="*70)
    
    print("\nCleaning narratives...")
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    
    # Remove any that became empty after cleaning
    df = df[df['cleaned_narrative'].str.len() > 0].copy()
    print(f"After cleaning and removing empty: {len(df):,}")
    
    # Recalculate word count
    df['cleaned_word_count'] = df['cleaned_narrative'].apply(count_words)
    
    print("\nCleaned Narrative Statistics:")
    print(df['cleaned_word_count'].describe())
    
    return df


def save_data(df, output_path='data/filtered_complaints.csv'):
    """Save cleaned dataset"""
    print("\n" + "="*70)
    print("SAVING CLEANED DATASET")
    print("="*70)
    
    # Select relevant columns
    columns_to_save = [
        'Complaint ID',
        'Product',
        'Sub-product',
        'Issue',
        'Sub-issue',
        'Consumer complaint narrative',
        'cleaned_narrative',
        'Company',
        'State',
        'Date received',
        'Date sent to company'
    ]
    
    # Filter to only columns that exist
    available_columns = [col for col in columns_to_save if col in df.columns]
    df_final = df[available_columns].copy()
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_file, index=False)
    
    print(f"\nCleaned dataset saved to: {output_file}")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"\nColumns saved: {list(df_final.columns)}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("FINAL DATASET SUMMARY")
    print("="*70)
    print(f"Total complaints: {len(df_final):,}")
    print(f"\nProduct distribution:")
    print(df_final['Product'].value_counts())
    
    return df_final


def main():
    """Main execution function"""
    # Load data
    df = load_data('data/complaints.csv')
    if df is None:
        return
    
    # Perform EDA
    df = perform_eda(df)
    
    # Filter data
    df_filtered = filter_data(df)
    
    # Clean data
    df_cleaned = clean_data(df_filtered)
    
    # Save data
    df_final = save_data(df_cleaned)
    
    print("\n" + "="*70)
    print("Task 1 completed successfully!")
    print("="*70)
    print("\nKey Findings:")
    print("1. The dataset contains complaints across multiple financial products")
    print("2. A significant portion of complaints include detailed narratives")
    print("3. Narrative lengths vary considerably, from very brief to very detailed")
    print("4. Text cleaning improved data quality by removing boilerplate and normalizing text")
    print("5. The filtered dataset maintains good distribution across target product categories")
    print("\nNext Step: Run src/create_vector_store.py for Task 2")


if __name__ == "__main__":
    main()
