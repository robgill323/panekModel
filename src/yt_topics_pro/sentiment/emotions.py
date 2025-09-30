# src/yt_topics_pro/sentiment/emotions.py
"""
Multi-label emotion analysis using Hugging Face Transformers.
"""
import logging
from typing import List

import polars as pl
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from yt_topics_pro.config import settings

logger = logging.getLogger(__name__)

_emotions_pipeline = None


def get_emotions_pipeline():
    """Initializes and caches the emotion analysis pipeline."""
    global _emotions_pipeline
    if _emotions_pipeline is None:
        logger.info(f"Loading emotions model: {settings.sentiment.emotions}")
        model_name = settings.sentiment.emotions
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if settings.gpu else -1
        _emotions_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,  # Return all labels
        )
    return _emotions_pipeline


def predict_emotions(texts: List[str], batch_size: int = 8) -> pl.DataFrame:
    """
    Predicts multi-label emotions for a list of texts.

    Args:
        texts: A list of text chunks.
        batch_size: Batch size for the pipeline.

    Returns:
        A Polars DataFrame with columns for each emotion and its score.
    """
    if not texts:
        return pl.DataFrame()

    pipe = get_emotions_pipeline()
    logger.info(f"Predicting emotions for {len(texts)} texts...")

    # The pipeline returns a list of lists of dicts
    results = pipe(texts, batch_size=batch_size)

    # Flatten and normalize the results
    processed_results = []
    for result_set in results:
        row = {item["label"]: item["score"] for item in result_set}
        processed_results.append(row)

    df = pl.from_records(processed_results)
    
    # Rename columns to be more descriptive
    df = df.rename({col: f"emotion_{col}" for col in df.columns})

    return df


if __name__ == "__main__":
    # Example Usage
    sample_texts = [
        "I am so excited for the party tonight!",
        "She felt a deep sense of remorse for her actions.",
        "The news filled him with anger and frustration.",
    ]
    emotions_df = predict_emotions(sample_texts)
    print("--- Emotion Predictions ---")
    print(emotions_df.select(pl.all().sorted(by=emotions_df.columns[0], descending=True)))
