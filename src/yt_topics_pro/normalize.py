# src/yt_topics_pro/normalize.py
"""
Functions for normalizing text data.
"""
import logging
import re
from typing import List

import polars as pl

logger = logging.getLogger(__name__)

# A basic list of English stopwords. For a real project, using a library like
# NLTK or spaCy would be more robust.
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves"
}


def _remove_repeated_phrases(text: str) -> str:
    """
    Removes consecutively repeated phrases from a string using Python's regex engine,
    which supports backreferences. This is a more robust implementation.
    e.g., "hello world hello world" -> "hello world"
    """
    if not isinstance(text, str):
        return text
    
    # This regex finds sequences of one or more words that are immediately repeated.
    # We loop `re.sub` until no more changes are made to handle cases like 
    # "phrase phrase phrase" -> "phrase phrase" -> "phrase".
    previous_text = ""
    # The regex: \b matches a word boundary, (\w+(\s+\w+)*) captures a sequence of words,
    # \s+ matches the space between repetitions, and \1 is a backreference to the captured phrase.
    # We make it case-insensitive to catch more variations.
    pattern = re.compile(r'\b(\w+(?:\s+\w+)*)\s+\1\b', re.IGNORECASE)
    
    while previous_text != text:
        previous_text = text
        text = pattern.sub(r'\1', text)
        
    return text


def normalize_text_series(text_series: pl.Series, stopwords: list[str]) -> pl.Series:
    """
    Applies a series of normalization steps to a Polars Series of text.
    The order of operations is critical to solving the repetition issue.
    """
    stopwords_set = set(stopwords)

    # Step 1: Apply the complex UDF for repeated phrases on the raw text FIRST.
    # This is crucial to catch the repetitions before any other processing
    # alters the text structure.
    series = text_series.map_elements(
        _remove_repeated_phrases, return_dtype=pl.String
    )

    # Step 2: Continue with the rest of the cleaning pipeline
    series = (
        series.str.replace_all(r"<[^>]+>", " ")
        .str.replace_all(r"c\d+c?", "")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z\s]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

    # Step 3: Final tokenization and stopword removal
    return (
        series.str.split(" ")
        .list.eval(
            pl.element().filter(
                ~pl.element().is_in(stopwords_set) & (pl.element() != "")
            )
        )
        .list.join(" ")
    )


def normalize_transcripts(
    transcripts_df: pl.DataFrame,
    stopwords: List[str] = None
) -> pl.DataFrame:
    """
    Normalizes the 'text' column of the transcripts DataFrame.

    Args:
        transcripts_df: DataFrame with a 'text' column.
        stopwords: A list of stopwords to remove. If None, a default list is used.

    Returns:
        A new DataFrame with a 'normalized_text' column.
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
        
    logger.info("Normalizing transcript text...")
    
    normalized_df = transcripts_df.with_columns(
        normalized_text=normalize_text_series(pl.col("text"), stopwords)
    )
    
    logger.info("Text normalization complete.")
    return normalized_df
