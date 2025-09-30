# src/yt_topics_pro/topics/evaluate.py
"""
Functions for evaluating the quality of a BERTopic model.
"""
import logging
from pathlib import Path

import polars as pl
from bertopic import BERTopic

from .. import config
from ..storage import tables

logger = logging.getLogger(__name__)


def evaluate_model(
    topic_model: BERTopic,
    docs: list[str],
    processed_chunks: pl.DataFrame
) -> pl.DataFrame:
    """
    Evaluates the topic model on various metrics.

    Args:
        topic_model: The fitted BERTopic model.
        docs: The list of documents used to train the model.
        processed_chunks: The DataFrame of processed chunks.

    Returns:
        A DataFrame containing evaluation metrics.
    """
    logger.info("Evaluating topic model...")
    
    # For this example, we'll just get the topic info DataFrame,
    # which contains basic stats like topic size and representation.
    # A full implementation would calculate coherence scores (e.g., C_v, NPMI)
    # and diversity metrics.
    
    topic_info_df = topic_model.get_topic_info()
    
    logger.info("Evaluation complete.")
    return pl.from_pandas(topic_info_df)


def run_evaluation():
    """
    Loads a trained model and data, runs evaluation, and saves the results.
    """
    logger.info("Starting model evaluation process...")
    
    # Load processed data
    processed_data = tables.load_from_parquet("processed")
    if "chunks" not in processed_data:
        logger.error("No processed data found. Please run `process` first.")
        return

    chunks_df = processed_data["chunks"].collect()
    docs = chunks_df["normalized_text"].to_list()

    # Load the BERTopic model
    model_path = Path(config.settings.storage.models_dir) / "bertopic_model"
    if not model_path.exists():
        logger.error(f"BERTopic model not found at {model_path}. Please run `process` first.")
        return
    
    logger.info(f"Loading BERTopic model from {model_path}...")
    topic_model = BERTopic.load(str(model_path))

    # Evaluate the model
    evaluation_results = evaluate_model(topic_model, docs, chunks_df)

    # Convert list columns to strings for CSV compatibility
    evaluation_results_for_csv = evaluation_results.with_columns(
        pl.col("Representation").list.join(", ").alias("Representation"),
        pl.col("Representative_Docs").list.join("\n---\n").alias("Representative_Docs"),
    )

    # Save results
    reports_dir = Path(config.settings.storage.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "topic_model_evaluation.csv"
    evaluation_results_for_csv.write_csv(results_path)
    
    logger.info(f"Evaluation results saved to {results_path}")
    print(evaluation_results)
