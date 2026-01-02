# Intelligent Complaint Analysis for Financial Services

A Retrieval-Augmented Generation (RAG) pipeline for analyzing consumer complaints from the Consumer Financial Protection Bureau (CFPB) database.

## Project Overview

This project implements a comprehensive RAG system to analyze and retrieve relevant consumer financial complaints. The system focuses on four key product categories:
- Credit cards
- Personal loans
- Savings accounts
- Money transfers

## Features

- **Exploratory Data Analysis**: Comprehensive analysis of CFPB complaint data
- **Data Preprocessing**: Text cleaning and normalization for improved embedding quality
- **Stratified Sampling**: Proportional representation across product categories
- **Text Chunking**: Intelligent text segmentation for optimal semantic search
- **Vector Embeddings**: State-of-the-art sentence transformers for text encoding
- **Vector Store**: FAISS-based efficient similarity search

## Project Structure

```
Intelligent-Complaint-Analysis/
├── .github/
│   └── workflows/          # CI/CD workflows
├── data/
│   ├── .gitkeep
│   ├── complaints.csv      # Raw CFPB data (not tracked)
│   └── filtered_complaints.csv  # Cleaned dataset (not tracked)
├── notebooks/
│   └── 01_eda_preprocessing.ipynb  # EDA and preprocessing notebook
├── src/
│   ├── download_data.py    # Data download utility
│   └── create_vector_store.py  # Vector store creation
├── vector_store/
│   ├── .gitkeep
│   ├── complaint_vectors.index  # FAISS index (not tracked)
│   ├── metadata.pkl        # Chunk metadata (not tracked)
│   └── config.pkl          # Configuration (not tracked)
├── tests/                  # Unit tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (for embedding model)
- ~2GB disk space for dataset and vector store

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Moggycool/Intelligent-Complaint-Analysis.git
   cd Intelligent-Complaint-Analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (if needed)
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage

### Task 1: Data Exploration and Preprocessing

#### Step 1: Download the CFPB Dataset

```bash
python src/download_data.py
```

This will download the official CFPB complaint dataset and save it to `data/complaints.csv`.

**Note**: The dataset is large (~2GB). Alternatively, download manually from:
https://www.consumerfinance.gov/data-research/consumer-complaints/

#### Step 2: Run EDA and Preprocessing

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
```

This notebook will:
- Perform exploratory data analysis
- Analyze product distribution
- Calculate narrative length statistics
- Filter data to target products
- Clean and normalize text
- Save the cleaned dataset to `data/filtered_complaints.csv`

**Key Findings**: The notebook provides visualizations and statistics about:
- Distribution of complaints across products
- Narrative length patterns (word count analysis)
- Proportion of complaints with/without narratives
- Effects of text cleaning on data quality

### Task 2: Vector Store Creation

After completing Task 1, create the vector store:

```bash
python src/create_vector_store.py
```

This script will:
1. Load the cleaned dataset
2. Create a stratified sample of 12,000 complaints (proportional across products)
3. Chunk texts using RecursiveCharacterTextSplitter (chunk_size=512, overlap=50)
4. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
5. Create a FAISS vector store
6. Save the vector store to `vector_store/` directory
7. Run a test search to verify functionality

**Sampling Strategy**: We use stratified sampling to ensure proportional representation across all product categories. With a sample size of 12,000 (middle of the 10k-15k range), we maintain the original distribution while making the dataset manageable for embedding.

**Chunking Approach**: 
- **Chunk Size**: 512 characters - balances semantic coherence with context preservation
- **Overlap**: 50 characters - ensures continuity across chunk boundaries
- **Splitter**: RecursiveCharacterTextSplitter with separators prioritizing natural boundaries (paragraphs, sentences, spaces)

**Embedding Model Choice**: `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight and efficient (80MB model size)
- 384-dimensional embeddings
- Excellent performance on semantic similarity tasks
- Fast inference time suitable for large-scale datasets
- Well-suited for complaint text domain

## Model Configuration

### Text Chunking Parameters

- `chunk_size`: 512 characters
- `chunk_overlap`: 50 characters
- `separators`: `["\n\n", "\n", ". ", " ", ""]`

### Embedding Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding Dimension**: 384
- **Max Sequence Length**: 256 tokens

### Vector Store

- **Type**: FAISS IndexFlatIP (Inner Product)
- **Similarity Metric**: Cosine similarity (via L2 normalization)

## Data Sources

- **CFPB Consumer Complaint Database**: https://www.consumerfinance.gov/data-research/consumer-complaints/
- **License**: Public Domain (U.S. Government Work)

## Dependencies

Key libraries used:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`, `seaborn`: Data visualization
- `sentence-transformers`: Text embeddings
- `faiss-cpu`: Vector similarity search
- `langchain`: Text chunking utilities
- `jupyter`: Interactive notebooks

See `requirements.txt` for complete list.

## Performance Metrics

After running the pipeline, you should see:
- **Dataset Size**: Variable (depends on CFPB updates)
- **Filtered Dataset**: Complaints from 4 product categories with narratives
- **Sample Size**: 12,000 complaints
- **Total Chunks**: ~25,000-30,000 (depending on narrative lengths)
- **Search Time**: <100ms for k=5 results

## Testing

Run tests with:
```bash
python -m pytest tests/
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in `create_vector_store.py`
   - Use a smaller sample size
   - Ensure you have at least 4GB available RAM

2. **Dataset Download Fails**
   - Download manually from CFPB website
   - Place in `data/complaints.csv`

3. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt` again

## Future Enhancements

- Integration with LLM for answer generation
- Web interface for complaint search
- Advanced filtering and faceted search
- Support for additional product categories
- Real-time complaint monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available under the MIT License. The CFPB data is in the public domain.

## Contact

For questions or feedback, please open an issue on GitHub.

## Acknowledgments

- Consumer Financial Protection Bureau for providing the complaint database
- Sentence-Transformers team for the embedding models
- FAISS team for the vector search library
