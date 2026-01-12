# Intelligent Complaint Analysis with RAG
## Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system for analyzing and answering questions over CFPB consumer complaint narratives.
It covers the full pipeline from exploratory data analysis and preprocessing through embedding, indexing, retrieval, generation, evaluation, and an interactive user interface.

The system is designed to be modular, reproducible, and production-oriented, with clear separation between data preparation, vector store construction, RAG pipeline logic, evaluation, and deployment.

## Project Objectives

- Perform transparent EDA and preprocessing on raw CFPB complaint data
- Construct a FAISS vector store from cleaned and sampled narratives
- Build an end-to-end RAG pipeline using modular components
- Evaluate RAG outputs via qualitative analysis
- Provide an interactive chat interface for real-time querying

## Repository Structure


```
Intelligent-Complaint-Analysis
├─ app.py
├─ notebooks
│  ├─ 01_task1_eda_and_preprocessing.ipynb
│  ├─ 02_task2_embedding_and_indexing.ipynb
│  ├─ 03_task3_rag_evaluation.ipynb
│  ├─ README.md
│  └─ __init__.py
├─ README.md
├─ requirements.txt
├─ run_rag_test.py
├─ src
│  ├─ chunking.py
│  ├─ data_loader.py
│  ├─ eda.py
│  ├─ embedding.py
│  ├─ evaluation.py
│  ├─ generator.py
│  ├─ preprocessing.py
│  ├─ prompts.py
│  ├─ rag_pipeline.py
│  ├─ retriever.py
│  ├─ sampling.py
│  ├─ vector_store.py
│  └─ __init__.py
├─ tests
│  ├─ run_rag_test.py
│  ├─ test_example.py
│  └─ __init__.py
└─ utils
   └─ paths.py

```
## Task 1: Exploratory Data Analysis & Preprocessing

Notebook: 01_task1_eda_and_preprocessing.ipynb

Key Steps
Load and inspect the full CFPB complaints dataset
Analyze:
- Narrative presence
- Product distribution
Narrative length statistics
Filter complaints:
- Retain only selected product categories
- Remove records without narratives
Apply text cleaning:
- Lowercasing
- Stopword removal
- Lemmatization
- Validate results using assertions
- Persist cleaned dataset for downstream tasks

Output
- data/processed/filtered_complaints.csv
This step ensures that only high-quality, semantically meaningful narratives feed the RAG pipeline.
## Task 2: Embedding Generation & Vector Store Construction

Notebook: 02_task2_embedding_and_indexing.ipynb
Key Steps
- Load cleaned data from Task 1
- Apply stratified sampling to preserve product balance
- Chunk narratives with metadata (complaint_id, product)
- Generate embeddings using:
   - sentence-transformers/all-MiniLM-L6-v2
- Validate embedding dimensions
- Build a FAISS index
- Persist vector store artifacts to disk

Output
- vector_store/index.faiss
- vector_store/metadata.pkl

These artifacts are directly consumed by the RAG pipeline in Task 3 and the interactive app in Task 4.

## Task 3: RAG Pipeline Construction & Evaluation

Notebook: 03_task3_rag_evaluation.ipynb
### Pipeline Components
1. Retriever
- FAISS similarity search
- Dynamic top-k retrieval based on query complexity

2. Generator

- google/flan-t5-base
- Deterministic generation settings for consistency

3. RAG Pipeline

- Context truncation
- Source attribution
- Modular orchestration

### Evaluation

- Run qualitative evaluation on 8 representative questions
- Inspect:
  - Answer relevance
  - Faithfulness to retrieved sources
  - Coverage of key complaint themes
- Export evaluation results for reporting

### Output

data/processed/task3_rag_qualitative_evaluation.csv

This task demonstrates correct wiring, execution, and evaluation of the RAG system using production modules.

## Task 4: Interactive Chat Interface

Application: app.py

An interactive Gradio-based chat interface allows users to query the complaint corpus in real time.

Features

Natural language question input

Retrieval-augmented answers

Source transparency (complaint metadata)

Uses the same FAISS vector store built in Task 2

Lightweight and CPU-friendly
### Run the App
- python app.py

## End-to-End Pipeline Summary
Raw CFPB Data
   ↓
EDA & Cleaning (Task 1)
   ↓
Stratified Sampling + Chunking (Task 2)
   ↓
Embedding Generation
   ↓
FAISS Vector Store (Persisted)
   ↓
RAG Retrieval + Generation (Task 3)
   ↓
Evaluation + Interactive Chat (Task 4)

## Requirements

Install dependencies with:
- pip install -r requirements.txt