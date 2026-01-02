# Getting Started Guide

This guide will help you set up and use the Intelligent Complaint Analysis RAG Agent.

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM and embeddings)
- 500MB free disk space (for dependencies and vector store)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Moggycool/Intelligent-Complaint-Analysis.git
cd Intelligent-Complaint-Analysis
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- LangChain for RAG orchestration
- FAISS/ChromaDB for vector storage
- OpenAI for LLM and embeddings
- Pandas for data processing
- Click for CLI interface

### 4. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

Required configuration:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 5. Run the Demo

Test that everything is working:

```bash
python demo.py
```

This will show you:
- Sample complaint data
- Available products
- Dataset statistics
- Example queries to try

## First Steps with the RAG Agent

### Initialize the Vector Store

Before asking questions, you need to build the vector store index:

```bash
python src/cli.py init --data-path data/sample_complaints.json
```

This process:
1. Loads complaint data
2. Creates embeddings using OpenAI
3. Builds FAISS index
4. Saves to disk for future use

**Note**: This step requires an OpenAI API key and will make API calls.

### Ask Your First Question

```bash
python src/cli.py ask "Why are people unhappy with Credit Cards?"
```

You should see:
- An AI-generated answer based on relevant complaints
- Source complaints used to generate the answer
- Metadata about the query

### Try Product Filtering

```bash
python src/cli.py ask "What are the main issues?" --product "Mortgage"
```

### Compare Multiple Products

```bash
python src/cli.py compare "What are the differences?" "Credit Card" "Mortgage"
```

### Interactive Mode

For an ongoing conversation:

```bash
python src/cli.py interactive
```

Commands in interactive mode:
- Type any question
- `products` - list available products
- `help` - show help
- `exit` or `quit` - exit

## Using Your Own Data

### Data Format

Your data should be in CSV or JSON format with these columns:

**Required:**
- `complaint_narrative` - The complaint text

**Recommended:**
- `product` - Product name
- `company` - Company name
- `state` - State code
- `issue` - Issue category

**Example JSON:**
```json
[
  {
    "complaint_narrative": "Your complaint text here...",
    "product": "Credit Card",
    "company": "XYZ Bank",
    "state": "CA",
    "issue": "Billing dispute"
  }
]
```

**Example CSV:**
```csv
complaint_narrative,product,company,state,issue
"Your complaint text here...","Credit Card","XYZ Bank","CA","Billing dispute"
```

### Loading Your Data

1. Place your data file in the `data/` directory
2. Update `.env` with the path:
   ```
   DATA_PATH=data/your_complaints.csv
   ```
3. Initialize the index:
   ```bash
   python src/cli.py init --force-rebuild
   ```

## Advanced Configuration

Edit `.env` to customize:

### Vector Store Options
```
VECTOR_STORE_TYPE=faiss  # or chromadb
VECTOR_STORE_PATH=vector_store/
```

### LLM Options
```
LLM_MODEL=gpt-3.5-turbo  # or gpt-4
LLM_TEMPERATURE=0.3      # 0.0-1.0 (lower = more focused)
LLM_MAX_TOKENS=500       # Max response length
```

### Retrieval Options
```
TOP_K_RESULTS=5          # Number of complaints to retrieve
CHUNK_SIZE=1000          # Text chunk size for embedding
```

### Embedding Options
```
EMBEDDING_MODEL=text-embedding-ada-002  # OpenAI embedding model
```

## Common Issues

### "OPENAI_API_KEY is required"

Make sure you:
1. Created a `.env` file
2. Added your API key
3. The key starts with `sk-`

### "Vector store not found"

Run initialization:
```bash
python src/cli.py init --data-path data/sample_complaints.json
```

### "No module named X"

Reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Out of Memory Errors

If processing large datasets:
1. Reduce `CHUNK_SIZE` in `.env`
2. Process data in batches
3. Use ChromaDB instead of FAISS (handles larger datasets)

## Performance Tips

### Speed Up Queries

1. **Use the saved index**: After first initialization, subsequent runs are fast
2. **Reduce TOP_K_RESULTS**: Fewer documents = faster generation
3. **Use gpt-3.5-turbo**: Faster and cheaper than GPT-4

### Improve Answer Quality

1. **Increase TOP_K_RESULTS**: More context for better answers
2. **Use gpt-4**: Higher quality reasoning
3. **Adjust LLM_TEMPERATURE**: Lower for more focused, higher for creative
4. **Clean your data**: Better input = better output

### Save API Costs

1. **Use gpt-3.5-turbo**: ~10x cheaper than GPT-4
2. **Reduce MAX_TOKENS**: Shorter responses = lower cost
3. **Cache results**: Save common queries
4. **Use smaller embeddings**: (if available in future)

## Next Steps

- Read the full [README](README.md) for detailed documentation
- Explore the source code in `src/rag_agent/`
- Customize prompts in `query_engine.py`
- Add more metadata fields for richer filtering
- Build a web interface on top of the CLI

## Getting Help

If you encounter issues:
1. Check this guide and the README
2. Review error messages carefully
3. Check your `.env` configuration
4. Verify your data format
5. Open an issue on GitHub

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ChromaDB Documentation](https://docs.trychroma.com/)
