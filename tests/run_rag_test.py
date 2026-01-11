""" Run a test of the RAG pipeline for question answering over customer complaint excerpts. """
from __future__ import annotations
from src.rag_pipeline import RAGPipeline
from src.generator import AnswerGenerator
from src.retriever import Retriever, dynamic_k
from utils.paths import VECTOR_STORE_DIR
import pickle
import faiss
from pathlib import Path
import sys
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ensure project root
_p = Path.cwd()
for _ancestor in [_p] + list(_p.parents):
    if (_ancestor / "src").exists() and (_ancestor / "utils").exists():
        sys.path.insert(0, str(_ancestor))
        break


index_path = os.path.join(str(VECTOR_STORE_DIR), "index.faiss")
metadata_path = os.path.join(str(VECTOR_STORE_DIR), "metadata.pkl")

print('Loading FAISS index from', index_path)
index = faiss.read_index(index_path)
with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)
print('Index size:', index.ntotal, 'Metadata len:', len(metadata))

retriever = Retriever(index=index, metadata=metadata,
                      model_name="sentence-transformers/all-MiniLM-L6-v2")

generator = AnswerGenerator(model_name="google/flan-t5-small",
                            device=-1, max_new_tokens=200, do_sample=False, temperature=0.0)

rag = RAGPipeline(retriever=retriever, generator=generator, k=5)

q = "What are customers complaining about regarding discrimination?"
print('Calling rag.answer...')
res = rag.answer(q)
print('\n=== ANSWER ===')
print(res.answer)
print('\n--- Sources (preview) ---')
for s in (res.sources or [])[:2]:
    print(s.get('complaint_id'), s.get('product'), s.get('score'))
    print((s.get('text') or '')[:300], '\n')
