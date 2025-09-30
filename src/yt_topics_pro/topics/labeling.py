# src/yt_topics_pro/topics/labeling.py
"""
Helper functions for generating and merging topic labels from different sources
like KeyBERT, YAKE, and LLMs.
"""
import logging
from typing import List, Dict

from keybert import KeyBERT
import yake

logger = logging.getLogger(__name__)


def generate_keybert_labels(
    topic_docs: Dict[int, List[str]], kw_model: KeyBERT
) -> Dict[int, List[str]]:
    """
    Generates topic labels using KeyBERT.

    Args:
        topic_docs: A dictionary mapping topic_id to a list of documents.
        kw_model: An initialized KeyBERT model.

    Returns:
        A dictionary mapping topic_id to a list of keyword labels.
    """
    logger.info("Generating labels with KeyBERT...")
    labels = {}
    for topic_id, docs in topic_docs.items():
        if topic_id == -1:
            continue
        combined_docs = " ".join(docs)
        keywords = kw_model.extract_keywords(
            combined_docs,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5,
        )
        labels[topic_id] = [kw[0] for kw in keywords]
    return labels


def generate_yake_labels(
    topic_docs: Dict[int, List[str]]
) -> Dict[int, List[str]]:
    """
    Generates topic labels using YAKE.

    Args:
        topic_docs: A dictionary mapping topic_id to a list of documents.

    Returns:
        A dictionary mapping topic_id to a list of keyword labels.
    """
    logger.info("Generating labels with YAKE...")
    labels = {}
    kw_extractor = yake.KeywordExtractor(
        lan="en", n=2, dedupLim=0.9, top=5, features=None
    )
    for topic_id, docs in topic_docs.items():
        if topic_id == -1:
            continue
        combined_docs = " ".join(docs)
        keywords = kw_extractor.extract_keywords(combined_docs)
        labels[topic_id] = [kw[0] for kw in keywords]
    return labels


def merge_labels(
    bertopic_labels: List[str],
    keybert_labels: Dict[int, List[str]],
    yake_labels: Dict[int, List[str]],
    llm_labels: Dict[int, str] = None,
) -> Dict[int, Dict[str, any]]:
    """
    Merges labels from different sources into a structured format.
    """
    logger.info("Merging labels from all sources...")
    final_labels = {}
    for topic_id, label in enumerate(bertopic_labels):
        # BERTopic's default labels are often just numbered words
        # e.g., "1_word1_word2_word3"
        final_labels[topic_id] = {
            "bertopic_default": label,
            "keybert": keybert_labels.get(topic_id, []),
            "yake": yake_labels.get(topic_id, []),
            "llm": llm_labels.get(topic_id, None) if llm_labels else None,
        }
    return final_labels

if __name__ == "__main__":
    print("Labeling module. Contains helpers for KeyBERT and YAKE.")
