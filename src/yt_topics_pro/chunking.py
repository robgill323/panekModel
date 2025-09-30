# src/yt_topics_pro/chunking.py
"""
Functions for chunking transcript segments into larger, meaningful blocks
for embedding and topic modeling.
"""
import logging
from typing import List, Dict, Any

import polars as pl

from yt_topics_pro.config import settings

logger = logging.getLogger(__name__)


def chunk_transcript(
    transcript_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Chunks transcript segments into windows of a target character count.

    This is a simplified chunker. A more advanced version would be
    sentence-aware, using a library like `nltk` to split on sentence
    boundaries while respecting the `max_chars` limit.

    Args:
        transcript_df: DataFrame with columns [video_id, text, start, duration].

    Returns:
        A DataFrame with chunked text and aggregated metadata.
        Columns: [video_id, chunk_id, text, start, duration, num_segments].
    """
    logger.info(
        f"Chunking transcripts with max_chars={settings.chunking.max_chars} "
        f"and overlap={settings.chunking.overlap}"
    )

    chunks = []
    for video_id, group in transcript_df.group_by("video_id"):
        # Use the 'normalized_text' column for chunking
        full_text = group["normalized_text"].str.join(" ").item()

        # Simple, non-overlapping chunking for demonstration
        text_chunks = [
            full_text[i : i + settings.chunking.max_chars]
            for i in range(0, len(full_text), settings.chunking.max_chars - settings.chunking.overlap)
        ]

        for i, chunk_text in enumerate(text_chunks):
            chunks.append(
                {
                    "video_id": video_id,
                    "chunk_id": i,
                    "normalized_text": chunk_text,
                    "start_time": group["start"].first(),  # Keep the first start_time
                }
            )

    if not chunks:
        return pl.DataFrame({
            "video_id": [], "chunk_id": [], "normalized_text": []
        }, schema={
            "video_id": pl.Utf8, "chunk_id": pl.Int64, "normalized_text": pl.Utf8
        })

    return pl.from_records(chunks)


if __name__ == "__main__":
    # Example Usage
    sample_transcript = pl.DataFrame({
        "video_id": ["v1"] * 5,
        "normalized_text": ["Hello world.", "This is a test.", "We are chunking.", "This is fun.", "The end."],
        "start": [0.1, 1.2, 2.5, 3.8, 5.0],
        "duration": [1.0, 1.1, 1.2, 1.1, 0.8],
    })
    
    settings.chunking.max_chars = 20 # Force multiple chunks
    chunked_df = chunk_transcript(sample_transcript)
    
    print("--- Original Transcript ---")
    print(sample_transcript)
    print("\n--- Chunked DataFrame ---")
    print(chunked_df)
    print(f"\nContent of first chunk:\n'{chunked_df[0, 'normalized_text']}'")
