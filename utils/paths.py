"""
Centralized path management utility.

This module standardizes file paths across the project to improve
portability, readability, and production readiness.
"""
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.pkl"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"
