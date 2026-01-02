# Quick Start Guide

This guide will help you get started with the Intelligent Complaint Analysis project quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM
- ~2GB disk space for dataset

## Installation

### Step 1: Clone and Setup

```bash
git clone https://github.com/Moggycool/Intelligent-Complaint-Analysis.git
cd Intelligent-Complaint-Analysis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Test with Sample Data

If you want to test the pipeline quickly without downloading the full CFPB dataset:

```bash
# Generate sample data (5,000 complaints)
python src/generate_sample_data.py

# Run EDA and preprocessing
python src/eda_preprocessing.py

# Create vector store
python src/create_vector_store.py
```

This will:
1. Create `data/complaints.csv` with sample data
2. Perform EDA and save filtered data to `data/filtered_complaints.csv`
3. Create vector store in `vector_store/` directory

## Full Pipeline with Real Data

### Option A: Using Scripts

```bash
# 1. Download CFPB data (~2GB, may take a few minutes)
python src/download_data.py

# 2. Run EDA and preprocessing
python src/eda_preprocessing.py

# 3. Create vector store
python src/create_vector_store.py
```

### Option B: Using Notebook

```bash
# 1. Download data
python src/download_data.py

# 2. Start Jupyter
jupyter notebook

# 3. Open and run notebooks/01_eda_preprocessing.ipynb

# 4. Create vector store
python src/create_vector_store.py
```

## Expected Output

After running the complete pipeline, you should have:

```
data/
  ├── complaints.csv           # Raw CFPB data (not in git)
  └── filtered_complaints.csv  # Cleaned data (not in git)

vector_store/
  ├── complaint_vectors.index  # FAISS index (not in git)
  ├── metadata.pkl            # Chunk metadata (not in git)
  └── config.pkl              # Configuration (not in git)
```

## Troubleshooting

### "ModuleNotFoundError"
Make sure you activated the virtual environment and installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Out of Memory" Error
- Reduce the sample size in `create_vector_store.py` (change `n_samples=12000` to a smaller value)
- Close other applications
- Use a machine with more RAM

### Download Timeout
The CFPB dataset is large. If download fails:
1. Download manually from: https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Save as `data/complaints.csv`
3. Continue with step 2

## What's Next?

After completing the pipeline:
- The vector store can be used for semantic search
- Integrate with LLMs for question answering
- Build a web interface for complaint search
- Add more analysis and visualizations

## Getting Help

- Check the full README.md for detailed documentation
- Review IMPLEMENTATION.md for technical details
- Open an issue on GitHub for bugs or questions
