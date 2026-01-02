"""
Utility script to download CFPB complaint dataset
"""
import requests
import pandas as pd
import os

def download_cfpb_data(output_path='data/complaints.csv'):
    """
    Download CFPB complaint dataset from the official source
    """
    # CFPB public complaints database URL
    url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
    
    print(f"Downloading CFPB complaint dataset from {url}...")
    
    try:
        # Download the file
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Save zip file temporarily
        zip_path = 'data/complaints.csv.zip'
        os.makedirs('data', exist_ok=True)
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Download complete. Extracting...")
        
        # Read the CSV from the zip file
        df = pd.read_csv(zip_path, compression='zip', low_memory=False)
        
        # Save as uncompressed CSV
        df.to_csv(output_path, index=False)
        
        # Remove zip file
        os.remove(zip_path)
        
        print(f"Dataset saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("You may need to download the dataset manually from:")
        print("https://www.consumerfinance.gov/data-research/consumer-complaints/")
        raise

if __name__ == "__main__":
    download_cfpb_data()
