# Intelligent Complaint Analysis for Financial Services

A RAG (Retrieval-Augmented Generation) agent that enables internal users to ask plain-English questions about customer complaints and receive intelligent, data-driven insights.

## Features

- 🔍 **Semantic Search**: Uses vector embeddings (FAISS or ChromaDB) to find relevant complaint narratives
- 🤖 **AI-Powered Analysis**: Leverages LLMs (OpenAI GPT) to generate concise, actionable insights
- 🏢 **Multi-Product Support**: Filter and compare complaints across different financial products
- 💬 **Natural Language Queries**: Ask questions in plain English, no technical knowledge required
- ⚡ **Fast Retrieval**: Efficient vector search for quick responses

## Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Moggycool/Intelligent-Complaint-Analysis.git
cd Intelligent-Complaint-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

4. Initialize the RAG agent with sample data:
```bash
python src/cli.py init --data-path data/sample_complaints.json
```

## Usage

### Command Line Interface

#### Ask a Question
```bash
python src/cli.py ask "Why are people unhappy with Credit Cards?"
```

#### Filter by Product
```bash
python src/cli.py ask "What are the main issues?" --product "Credit Card"
```

#### Compare Products
```bash
python src/cli.py compare "What are the main differences in customer complaints?" "Credit Card" "Mortgage" "Checking Account"
```

#### List Available Products
```bash
python src/cli.py products
```

#### Interactive Mode
```bash
python src/cli.py interactive
```

### Python API

```python
from src.rag_agent.agent import ComplaintRAGAgent

# Initialize the agent
agent = ComplaintRAGAgent(data_path="data/sample_complaints.json")
agent.initialize()

# Ask a question
result = agent.ask("Why are customers frustrated with credit cards?")
print(result['answer'])

# Filter by product
result = agent.ask(
    "What are the main issues?",
    product="Mortgage"
)

# Compare products
result = agent.compare(
    "Compare complaint patterns",
    products=["Credit Card", "Checking Account"]
)
```

## Architecture

The RAG agent consists of four main components:

1. **Data Loader** (`data_loader.py`): Loads and preprocesses complaint narratives from CSV or JSON files
2. **Vector Store** (`vector_store.py`): Creates embeddings and manages FAISS/ChromaDB for semantic search
3. **Query Engine** (`query_engine.py`): Retrieves relevant complaints and generates answers using LLM
4. **Agent** (`agent.py`): Orchestrates the entire pipeline and provides a simple interface

## Configuration

Edit `.env` file to customize:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `VECTOR_STORE_TYPE`: Choose `faiss` or `chromadb` (default: faiss)
- `LLM_MODEL`: GPT model to use (default: gpt-3.5-turbo)
- `TOP_K_RESULTS`: Number of complaints to retrieve (default: 5)

See `.env.example` for all available options.

## Data Format

The agent expects complaint data in CSV or JSON format with these fields:

- `complaint_narrative` (required): The complaint text
- `product` (recommended): Product name (e.g., "Credit Card", "Mortgage")
- `company`: Company name
- `state`: State abbreviation
- `issue`: Issue category

Example JSON:
```json
[
  {
    "complaint_narrative": "I was charged unauthorized fees...",
    "product": "Credit Card",
    "company": "ABC Bank",
    "state": "CA",
    "issue": "Fraudulent charges"
  }
]
```

## Example Queries

- "Why are people unhappy with Credit Cards?"
- "What are the most common mortgage complaints?"
- "How do checking account issues compare to savings account issues?"
- "What problems are customers having with loan applications?"
- "Why are customers frustrated with customer service?"

## Supported Products

The sample dataset includes:
- Credit Card
- Mortgage
- Checking Account
- Savings Account
- Personal Loan
- Student Loan

(Your actual products will depend on your data)

## Vector Store

The agent builds a vector index from your complaint data:

- **First run**: Builds and saves the index (takes a few minutes)
- **Subsequent runs**: Loads the existing index (fast)
- **Rebuild**: Use `--force-rebuild` flag to rebuild from scratch

## Development

### Project Structure
```
Intelligent-Complaint-Analysis/
├── src/
│   ├── rag_agent/
│   │   ├── __init__.py
│   │   ├── agent.py          # Main RAG agent
│   │   ├── config.py         # Configuration management
│   │   ├── data_loader.py    # Data loading & preprocessing
│   │   ├── vector_store.py   # Vector store management
│   │   └── query_engine.py   # RAG query engine
│   └── cli.py                # Command-line interface
├── data/
│   └── sample_complaints.json
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

### Adding Your Own Data

1. Prepare your complaint data in CSV or JSON format
2. Update `DATA_PATH` in `.env` or use `--data-path` flag
3. Run `python src/cli.py init --force-rebuild`

## Troubleshooting

**"OPENAI_API_KEY is required"**
- Make sure you've created a `.env` file and added your API key

**"No such file or directory: data/complaints.csv"**
- Use the sample data: `--data-path data/sample_complaints.json`
- Or provide your own data file

**"Vector store not found"**
- Run `python src/cli.py init` to build the index first

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
