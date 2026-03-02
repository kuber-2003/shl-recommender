"""
Build FAISS vector index from scraped SHL assessments.
Run AFTER scrape_catalog.py.
Output: ../data/faiss_index.bin + ../data/assessments_meta.pkl
"""

import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
ASSESSMENTS_FILE = os.path.join(DATA_DIR, "shl_assessments.json")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
META_FILE = os.path.join(DATA_DIR, "assessments_meta.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality, 384-dim


def build_document(assessment: dict) -> str:
    """
    Create a rich text document from assessment fields for embedding.
    More context = better semantic retrieval.
    """
    parts = []

    if assessment.get("name"):
        parts.append(f"Assessment: {assessment['name']}")

    if assessment.get("description"):
        parts.append(f"Description: {assessment['description']}")

    if assessment.get("test_type"):
        types_str = ", ".join(assessment["test_type"])
        parts.append(f"Test Types: {types_str}")

    if assessment.get("duration"):
        parts.append(f"Duration: {assessment['duration']} minutes")

    if assessment.get("remote_support"):
        parts.append(f"Remote Testing: {assessment['remote_support']}")

    if assessment.get("adaptive_support"):
        parts.append(f"Adaptive Testing: {assessment['adaptive_support']}")

    return " | ".join(parts)


def build_index():
    print("=" * 60)
    print("Building FAISS Vector Index")
    print("=" * 60)

    # Load assessments
    with open(ASSESSMENTS_FILE, "r") as f:
        assessments = json.load(f)

    print(f"\nLoaded {len(assessments)} assessments")

    # Build documents
    documents = [build_document(a) for a in assessments]

    # Embed
    print(f"\nLoading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding assessments...")
    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # For cosine similarity via inner product
    )

    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    print(f"Embedding shape: {embeddings.shape}, dim={dim}")

    # Build FAISS index (Inner Product = cosine similarity when normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")

    # Save index
    faiss.write_index(index, INDEX_FILE)
    print(f"Index saved → {INDEX_FILE}")

    # Save metadata (assessments list + model name)
    meta = {
        "assessments": assessments,
        "documents": documents,
        "model_name": MODEL_NAME,
    }
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    print(f"Metadata saved → {META_FILE}")

    print("\n✅ Index build complete!")


if __name__ == "__main__":
    build_index()
