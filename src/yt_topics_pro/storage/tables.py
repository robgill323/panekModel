# src/yt_topics_pro/storage/tables.py
"""
Defines Parquet schemas and provides helpers for writing data.
"""
import logging
import os
from pathlib import Path

import polars as pl

from yt_topics_pro.config import settings

logger = logging.getLogger(__name__)


def save_to_parquet(data: dict[str, pl.DataFrame], table_group: str):
    """
    Saves a dictionary of DataFrames to Parquet files in a subdirectory.

    Args:
        data: Dictionary where keys are table names and values are DataFrames.
        table_group: The name of the processing stage (e.g., 'raw', 'processed').
    """
    output_dir = Path(settings.storage.parquet_dir) / table_group
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        path = output_dir / f"{name}.parquet"
        logger.info(f"Writing {df.shape[0]} rows to {path}")
        try:
            df.write_parquet(path)
        except Exception as e:
            logger.error(f"Failed to write Parquet file {path}: {e}")


def load_from_parquet(table_group: str) -> dict[str, pl.LazyFrame]:
    """
    Loads all Parquet files from a subdirectory as LazyFrames.

    Args:
        table_group: The name of the processing stage (e.g., 'raw', 'processed').

    Returns:
        A dictionary of table names to Polars LazyFrames.
    """
    input_dir = Path(settings.storage.parquet_dir) / table_group
    if not input_dir.exists():
        logger.warning(f"Parquet directory not found: {input_dir}")
        return {}

    lazy_frames = {}
    for path in input_dir.glob("*.parquet"):
        table_name = path.stem
        logger.info(f"Loading lazy frame for {table_name} from {path}")
        lazy_frames[table_name] = pl.scan_parquet(path)

    return lazy_frames


if __name__ == "__main__":
    # Example Usage
    # Create dummy data
    raw_data = {
        "transcripts": pl.DataFrame({"video_id": ["v1"], "text": ["hello"]}),
        "metadata": pl.DataFrame({"video_id": ["v1"], "title": ["title1"]}),
    }
    processed_data = {
        "chunks": pl.DataFrame({"chunk_id": [1], "text": ["chunk1"]}),
    }

    # Save
    save_to_parquet(raw_data, "raw")
    save_to_parquet(processed_data, "processed")

    # Load
    loaded_raw = load_from_parquet("raw")
    loaded_processed = load_from_parquet("processed")

    print("--- Loaded Raw LazyFrames ---")
    print(loaded_raw)
    print(loaded_raw["transcripts"].fetch(5))

    print("\n--- Loaded Processed LazyFrames ---")
    print(loaded_processed)
    print(loaded_processed["chunks"].fetch(5))

    # Clean up
    import shutil
    if os.path.exists(settings.storage.parquet_dir):
        shutil.rmtree(settings.storage.parquet_dir)
