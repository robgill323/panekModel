# src/yt_topics_pro/storage/faiss_index.py
"""
Functions for building, saving, loading, and searching a FAISS vector index.
"""
import logging
from pathlib import Path

import numpy as np
import faiss

from yt_topics_pro.config import settings

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index from a set of embeddings.

    Args:
        embeddings: A 2D numpy array of document embeddings.

    Returns:
        A trained FAISS index.
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]  # Embedding dimension
    index_type = "IndexFlatL2" # Simple L2 distance index
    
    logger.info(f"Building FAISS index of type '{index_type}' with {embeddings.shape[0]} vectors of dimension {d}.")

    if settings.gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, d)
            logger.info("Using GPU for FAISS index.")
        except AttributeError:
            logger.warning("FAISS GPU extensions not found. Falling back to CPU.")
            index = faiss.index_factory(d, index_type)
    else:
        index = faiss.index_factory(d, index_type)

    index.add(embeddings)
    logger.info(f"FAISS index built. Is trained: {index.is_trained}, Total vectors: {index.ntotal}")
    return index


def save_faiss_index(index: faiss.Index, name: str = "main_index"):
    """Saves a FAISS index to disk."""
    output_dir = Path(settings.storage.faiss_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.faiss"
    logger.info(f"Saving FAISS index to {path}")
    faiss.write_index(index, str(path))


def load_faiss_index(name: str = "main_index") -> faiss.Index:
    """Loads a FAISS index from disk."""
    path = Path(settings.storage.faiss_dir) / f"{name}.faiss"
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at {path}")
    
    logger.info(f"Loading FAISS index from {path}")
    if settings.gpu:
        try:
            index = faiss.read_index(str(path))
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
            logger.info("Successfully loaded FAISS index to GPU.")
            return gpu_index
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"Could not load FAISS to GPU ({e}). Loading to CPU.")
            return faiss.read_index(str(path))
    else:
        return faiss.read_index(str(path))


def search_faiss_index(index: faiss.Index, query_vectors: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Searches the index for the top k nearest neighbors.

    Args:
        index: The FAISS index to search.
        query_vectors: A 2D numpy array of query embeddings.
        k: The number of neighbors to retrieve.

    Returns:
        A tuple of (distances, indices).
    """
    if query_vectors.dtype != np.float32:
        query_vectors = query_vectors.astype(np.float32)
        
    logger.info(f"Searching for {k} nearest neighbors for {query_vectors.shape[0]} queries.")
    distances, indices = index.search(query_vectors, k)
    return distances, indices


if __name__ == "__main__":
    # Example Usage
    # 1. Create dummy data
    d = 64  # dimension
    nb = 1000 # database size
    nq = 10 # nb of queries
    np.random.seed(1234)
    db_vectors = np.random.random((nb, d)).astype('float32')
    query_vectors = np.random.random((nq, d)).astype('float32')

    # 2. Build and save index
    index = build_faiss_index(db_vectors)
    save_faiss_index(index, "test_index")

    # 3. Load index
    loaded_index = load_faiss_index("test_index")
    print(f"Index loaded. Total vectors: {loaded_index.ntotal}")

    # 4. Search
    distances, indices = search_faiss_index(loaded_index, query_vectors, k=4)
    print("\n--- Search Results (Indices) ---")
    print(indices)
    print("\n--- Search Results (Distances) ---")
    print(distances)

    # Clean up
    import os
    os.remove(Path(settings.storage.faiss_dir) / "test_index.faiss")
