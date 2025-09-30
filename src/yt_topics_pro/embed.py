# src/yt_topics_pro/embed.py
"""
Handles loading of SentenceTransformer models and embedding of text chunks.
"""
import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from yt_topics_pro.config import settings

logger = logging.getLogger(__name__)

_model_cache = {}


def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model specified in settings.
    """
    model_name = settings.embedding_model
    if model_name in _model_cache:
        return _model_cache[model_name]

    logger.info(f"Loading embedding model: {model_name}")
    device = "cuda" if settings.gpu and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    logger.info(f"Using device: {model.device}")

    if settings.gpu and model.device.type == "cuda":
        # FP16 is not always a win, depends on GPU architecture.
        # For modern GPUs (Ampere+), it's generally faster.
        logger.info("Attempting to use FP16 for embeddings.")
        # This is a simplified way; for robust FP16, you might need more.
        # model.half() # This can sometimes cause issues.
        # A better approach is to use `model.encode` with `convert_to_tensor=True`
        # and then `.half()` on the tensor, but BERTopic handles encoding internally.
        pass

    _model_cache[model_name] = model
    return model


def embed_texts(texts: List[str], batch_size: int = 32) -> torch.Tensor:
    """
    Embeds a list of texts using the configured model.

    Args:
        texts: A list of strings to embed.
        batch_size: The batch size for encoding.

    Returns:
        A torch.Tensor containing the embeddings.
    """
    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    logger.info(f"Generated embeddings of shape: {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    # Example Usage
    sample_texts = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]

    # Test with default model
    embeddings_tensor = embed_texts(sample_texts)
    print(f"--- Embeddings Tensor (shape: {embeddings_tensor.shape}) ---")
    print(embeddings_tensor)

    # Test with a different model
    settings.embedding_model = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    _model_cache.clear() # Clear cache to load new model
    embeddings_tensor_mini = embed_texts(sample_texts)
    print(f"\n--- MiniLM Embeddings Tensor (shape: {embeddings_tensor_mini.shape}) ---")
    print(embeddings_tensor_mini)
