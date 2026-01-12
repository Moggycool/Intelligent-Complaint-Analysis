## End-to-End Data → RAG Pipeline

1. `01_task1_eda_and_preprocessing.ipynb`
   - Explores, cleans, filters, and validates CFPB complaint narratives

2. `02_task2_embedding_and_indexing.ipynb`
   - Performs stratified sampling
   - Chunks narratives with metadata
   - Generates embeddings
   - Builds and persists a FAISS vector store

3. `03_task3_rag_evaluation.ipynb`
   - Loads the persisted FAISS index and metadata
   - Performs semantic retrieval over the prepared corpus
4. 
