# src/yt_topics_pro/sentiment/aggregate.py
"""
Functions to aggregate sentiment and emotion scores by topic, video, or channel.
"""
import logging

import polars as pl

logger = logging.getLogger(__name__)


def aggregate_sentiment_by_topic(
    results_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregates sentiment and emotion scores per topic.

    Args:
        results_df: A DataFrame containing chunk-level data including
                    'topic', polarity scores, and emotion scores.

    Returns:
        A DataFrame with aggregated sentiment metrics per topic.
    """
    if "topic" not in results_df.columns:
        raise ValueError("Input DataFrame must contain a 'topic' column.")

    logger.info("Aggregating sentiment scores by topic...")

    # Identify polarity and emotion columns
    polarity_cols = [c for c in results_df.columns if c.startswith("polarity_")]
    emotion_cols = [c for c in results_df.columns if c.startswith("emotion_")]

    if not polarity_cols and not emotion_cols:
        logger.warning("No sentiment or emotion columns found for aggregation.")
        return pl.DataFrame({"topic": results_df["topic"].unique()})

    # Define aggregations
    aggs = []
    if polarity_cols:
        aggs.extend([pl.mean(col).alias(f"{col}_mean") for col in polarity_cols])
    if emotion_cols:
        aggs.extend([pl.mean(col).alias(f"{col}_mean") for col in emotion_cols])

    # Perform aggregation
    topic_sentiment = (
        results_df.lazy()
        .group_by("topic")
        .agg(aggs)
        .sort("topic")
        .collect()
    )

    logger.info("Sentiment aggregation complete.")
    return topic_sentiment


if __name__ == "__main__":
    # Example Usage
    sample_data = pl.DataFrame({
        "chunk_id": [1, 2, 3, 4],
        "topic": [0, 1, 0, 1],
        "polarity_positive": [0.9, 0.1, 0.8, 0.2],
        "polarity_negative": [0.1, 0.9, 0.2, 0.8],
        "emotion_joy": [0.8, 0.0, 0.7, 0.1],
        "emotion_sadness": [0.0, 0.8, 0.1, 0.7],
    })

    aggregated_df = aggregate_sentiment_by_topic(sample_data)
    print("--- Original Data ---")
    print(sample_data)
    print("\n--- Aggregated Sentiment by Topic ---")
    print(aggregated_df)
