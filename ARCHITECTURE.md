# Architecture Overview

This document explains the architecture and design of the Intelligent Complaint Analysis RAG Agent.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│  (CLI, Interactive Mode, or Python API)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    ComplaintRAGAgent                         │
│  - Orchestrates the entire pipeline                         │
│  - Manages initialization and lifecycle                      │
└────────┬─────────────────────────┬──────────────────────────┘
         │                         │
         ▼                         ▼
┌──────────────────┐    ┌───────────────────────────┐
│  Data Loader     │    │   Vector Store Manager     │
│  - Load CSV/JSON │    │   - Create embeddings      │
│  - Preprocess    │───▶│   - Build FAISS/Chroma     │
│  - Extract docs  │    │   - Semantic search        │
└──────────────────┘    └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   RAG Query Engine       │
                        │   - Retrieve context     │
                        │   - Format prompts       │
                        │   - Generate answers     │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │   OpenAI LLM             │
                        │   - GPT-3.5/GPT-4        │
                        │   - Generate insights    │
                        └──────────────────────────┘
```

## Core Components

### 1. Data Loader (`data_loader.py`)

**Purpose**: Load and preprocess complaint data from various sources.

**Key Responsibilities**:
- Load data from CSV or JSON files
- Clean and normalize text
- Extract metadata (product, company, state, issue)
- Convert to document format for vector store

**Key Methods**:
- `load_data()`: Read from file
- `preprocess()`: Clean and normalize
- `get_documents()`: Convert to vector store format
- `get_products()`: Extract unique product list

**Design Decisions**:
- Uses pandas for efficient data handling
- Flexible column mapping (supports various data formats)
- Text cleaning preserves important information while removing noise

### 2. Vector Store Manager (`vector_store.py`)

**Purpose**: Manage vector embeddings and semantic search.

**Key Responsibilities**:
- Create embeddings using OpenAI's embedding model
- Build and manage FAISS or ChromaDB vector store
- Perform semantic similarity search
- Support metadata filtering

**Key Methods**:
- `create_vector_store()`: Build index from documents
- `save_vector_store()`: Persist to disk
- `load_vector_store()`: Load from disk
- `search()`: Semantic similarity search
- `search_with_scores()`: Search with relevance scores

**Design Decisions**:
- Supports both FAISS (in-memory, fast) and ChromaDB (persistent, scalable)
- Embeddings cached on disk to avoid re-computation
- Metadata filtering for product-specific queries

### 3. RAG Query Engine (`query_engine.py`)

**Purpose**: Implement the RAG pattern - retrieve relevant context and generate answers.

**Key Responsibilities**:
- Retrieve relevant complaints using vector search
- Format context for LLM prompts
- Generate answers using OpenAI LLM
- Support single-product and multi-product queries

**Key Methods**:
- `query()`: Answer a question about complaints
- `compare_products()`: Compare across products
- `_format_context()`: Format retrieved documents

**Design Decisions**:
- Template-based prompts for consistency
- Structured output with sources and metadata
- Separate prompts for analysis vs comparison

### 4. Main Agent (`agent.py`)

**Purpose**: Orchestrate the entire pipeline and provide a simple API.

**Key Responsibilities**:
- Initialize all components
- Manage vector store lifecycle
- Provide simple query interface
- Handle configuration and error cases

**Key Methods**:
- `initialize()`: Set up the agent
- `ask()`: Ask a question
- `compare()`: Compare products
- `get_available_products()`: List products
- `rebuild_index()`: Rebuild vector store

**Design Decisions**:
- Lazy initialization of LLM to save API calls
- Automatic vector store detection and loading
- Clear separation of concerns

### 5. Configuration (`config.py`)

**Purpose**: Centralized configuration management.

**Features**:
- Environment variable support via python-dotenv
- Sensible defaults
- Configuration validation
- Type safety

### 6. CLI Interface (`cli.py`)

**Purpose**: User-friendly command-line interface.

**Commands**:
- `init`: Initialize vector store
- `ask`: Ask a question
- `compare`: Compare products
- `products`: List available products
- `interactive`: Interactive mode

**Design Decisions**:
- Click framework for robust CLI
- Colorized output for better UX
- Progress indicators
- Helpful error messages

## Data Flow

### Initialization Flow

```
1. User runs: python src/cli.py init
2. ComplaintRAGAgent.initialize()
3. ComplaintLoader loads and preprocesses data
4. VectorStoreManager creates embeddings (OpenAI API)
5. Vector store built and saved to disk
6. Products list cached
7. Ready for queries
```

### Query Flow

```
1. User asks: "Why are people unhappy with Credit Cards?"
2. ComplaintRAGAgent.ask()
3. VectorStoreManager.search() - semantic search
4. Top K relevant complaints retrieved
5. RAGQueryEngine formats context
6. LLM generates answer (OpenAI API)
7. Result returned with sources
8. CLI displays formatted output
```

### Comparison Flow

```
1. User compares: "Credit Card" vs "Mortgage"
2. ComplaintRAGAgent.compare()
3. For each product:
   - VectorStoreManager.search() with product filter
   - Retrieve top K complaints
4. RAGQueryEngine formats multi-product context
5. LLM generates comparative analysis
6. Result returned with all sources
7. CLI displays formatted output
```

## Technology Choices

### Why LangChain?

- **Abstraction**: Simplifies RAG implementation
- **Flexibility**: Easy to swap LLMs or vector stores
- **Community**: Large ecosystem and examples
- **Features**: Built-in document handling, prompts, chains

### Why FAISS?

- **Speed**: Extremely fast similarity search
- **Memory Efficient**: Optimized for large datasets
- **Mature**: Battle-tested by Facebook AI
- **Free**: No licensing costs

**Alternative**: ChromaDB for persistence and easier scaling

### Why OpenAI?

- **Quality**: Best-in-class LLM performance
- **Embeddings**: High-quality text embeddings
- **Reliability**: Production-ready API
- **Developer Experience**: Excellent documentation

**Future**: Could add support for open-source LLMs (Llama, Mistral)

## Design Patterns

### 1. Dependency Injection

Components receive dependencies via constructor:
```python
RAGQueryEngine(vector_store_manager)
```

**Benefits**: Testability, flexibility, clear dependencies

### 2. Strategy Pattern

Vector store type selected via configuration:
```python
if store_type == 'faiss':
    # Use FAISS
elif store_type == 'chromadb':
    # Use ChromaDB
```

**Benefits**: Easy to add new vector stores

### 3. Template Method

Query processing follows a template:
1. Retrieve documents
2. Format context
3. Generate answer
4. Return structured result

**Benefits**: Consistency, maintainability

### 4. Facade Pattern

`ComplaintRAGAgent` provides simple interface hiding complexity:
```python
agent.ask("question")  # Simple API
# Internally: load data, search, format, generate
```

**Benefits**: Ease of use, encapsulation

## Performance Considerations

### Embedding Creation

- **First run**: Slow (API calls for each document)
- **Subsequent runs**: Fast (loads cached embeddings)
- **Optimization**: Batch processing for large datasets

### Vector Search

- **FAISS**: Sub-millisecond for thousands of documents
- **ChromaDB**: Slightly slower but handles millions
- **Optimization**: Reduce dimensionality if needed

### LLM Generation

- **Bottleneck**: Network latency to OpenAI
- **Typical**: 1-5 seconds per query
- **Optimization**: Use GPT-3.5 instead of GPT-4

### Memory Usage

- **FAISS**: Entire index in memory
- **ChromaDB**: Disk-backed
- **Typical**: 100MB for 10,000 documents

## Security Considerations

### API Keys

- Stored in `.env` file (not committed to git)
- Loaded via environment variables
- Never logged or exposed

### Data Privacy

- All data processed locally
- Only embeddings sent to OpenAI
- No complaint text stored by OpenAI (per API terms)

### Input Validation

- File path validation
- Type checking on inputs
- Safe deserialization (controlled)

## Extensibility

### Adding New Data Sources

1. Extend `ComplaintLoader`
2. Implement format-specific parsing
3. Ensure consistent document format

### Adding New Vector Stores

1. Add new case in `VectorStoreManager`
2. Implement create, save, load methods
3. Update configuration

### Adding New LLM Providers

1. Update `RAGQueryEngine` LLM initialization
2. Handle provider-specific APIs
3. Update configuration

### Custom Prompts

1. Modify prompt templates in `query_engine.py`
2. Add new query methods for specialized tasks
3. Expose via CLI or API

## Testing Strategy

### Unit Tests

- Test individual components in isolation
- Mock external dependencies (OpenAI API)
- Cover edge cases and error handling

### Integration Tests

- Test component interactions
- Use test data files
- Verify end-to-end flows

### Manual Testing

- Real queries with sample data
- Performance testing with larger datasets
- User acceptance testing

## Future Enhancements

### Potential Features

1. **Web Interface**: Flask/FastAPI REST API
2. **Caching**: Cache common queries
3. **Analytics**: Track query patterns
4. **Multi-tenancy**: Support multiple datasets
5. **Fine-tuning**: Custom embeddings or LLM
6. **Feedback Loop**: User ratings to improve results
7. **Advanced Filtering**: Date ranges, severity, etc.
8. **Export**: Generate reports from queries
9. **Real-time**: Stream answers as they generate
10. **Visualization**: Charts and graphs from data

### Scalability Path

1. **Horizontal Scaling**: Multiple API instances
2. **Vector Store**: Migrate to Pinecone or Weaviate
3. **Caching Layer**: Redis for query results
4. **Async**: Parallel document processing
5. **Batch Processing**: Handle large data imports

## Conclusion

This architecture provides:
- **Simplicity**: Easy to understand and use
- **Flexibility**: Easy to extend and customize
- **Performance**: Fast queries and efficient resource use
- **Reliability**: Robust error handling and validation
- **Maintainability**: Clean code and clear separation of concerns

The modular design allows for incremental improvements without major refactoring.
