# src/yt_topics_pro/sentiment/polarity.py
"""
Sentiment polarity analysis using Hugging Face Transformers.
"""
import logging
from typing import List

import polars as pl
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from yt_topics_pro.config import settings

logger = logging.getLogger(__name__)

_sentiment_pipeline = None


def get_polarity_pipeline():
    """Initializes and caches the sentiment polarity pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info(f"Loading sentiment polarity model: {settings.sentiment.polarity}")
        model_name = settings.sentiment.polarity
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 0 if settings.gpu else -1
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True, # Get scores for all labels
        )
    return _sentiment_pipeline


def predict_polarity(texts: List[str], batch_size: int = 8) -> pl.DataFrame:
    """
    Predicts sentiment polarity (positive, neutral, negative) for a list of texts.

    Args:
        texts: A list of text chunks.
        batch_size: Batch size for the pipeline.

    Returns:
        A Polars DataFrame with columns for each sentiment label and its score.
    """
    if not texts:
        return pl.DataFrame()

    pipe = get_polarity_pipeline()
    logger.info(f"Predicting polarity for {len(texts)} texts...")
    
    # The pipeline returns a list of lists of dicts
    # e.g., [[{'label': 'positive', 'score': 0.9}, ...], ...]
    results = pipe(texts, batch_size=batch_size)

    # Flatten and normalize the results
    processed_results = []
    for result_set in results:
        row = {item["label"]: item["score"] for item in result_set}
        processed_results.append(row)

    df = pl.from_records(processed_results)
    
    # Rename columns to be more descriptive and calculate a single sentiment score
    df = df.rename({col: f"polarity_{col}" for col in df.columns})
    
    if "polarity_positive" in df.columns and "polarity_negative" in df.columns:
        df = df.with_columns(
            (pl.col("polarity_positive") - pl.col("polarity_negative")).alias("sentiment")
        )
    
    return df


if __name__ == "__main__":
    # Example Usage
    sample_texts = [
        "This is a wonderful movie, I loved it!",
        "The weather is okay today.",
        "I am not happy with the service provided.",
    ]
    polarity_df = predict_polarity(sample_texts)
    print("--- Polarity Predictions ---")
    print(polarity_df)
