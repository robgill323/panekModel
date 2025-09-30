# src/yt_topics_pro/topics/bertopic_runner.py
"""
Core BERTopic modeling pipeline.
"""
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import polars as pl
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP

from yt_topics_pro.config import settings
from yt_topics_pro.embed import get_embedding_model
from .. import config

logger = logging.getLogger(__name__)


def _get_bertopic_model(docs: List[str]) -> BERTopic:
    """
    Initializes and returns a BERTopic model with configured settings.
    """
    # Configure vectorizer to handle short documents
    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=2, # Ignore terms that appear in only one document
        ngram_range=(1, 2),
    )

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=10, # Lowered from 15 to find more local structure
        n_components=5, # Standard for BERTopic
        min_dist=0.0,
        metric='cosine',
        random_state=config.settings.seed
    )

    # Configure HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=5, # Lowered from 10 to allow smaller topics
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # Use settings from the config file
    bertopic_settings = config.settings.bertopic

    return BERTopic(
        language="english",
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=bertopic_settings.nr_topics,
        calculate_probabilities=bertopic_settings.calculate_probabilities,
        verbose=True,
    )


def run_bertopic(
    docs: List[str],
    embeddings: "np.ndarray",
    timestamps: List[str],
) -> Tuple[BERTopic, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Fits a BERTopic model and generates topics, probabilities, and topics over time.

    Args:
        docs: The text chunks to model.
        embeddings: Pre-computed embeddings for the docs.
        timestamps: Timestamps for each doc (for topics-over-time).

    Returns:
        A tuple containing:
        - The fitted BERTopic model.
        - A DataFrame of topics and their probabilities for each doc.
        - A DataFrame of topics over time.
        - A DataFrame for hierarchical topics.
    """
    logger.info("Running BERTopic pipeline...")
    topic_model = _get_bertopic_model(docs)

    logger.info("Fitting BERTopic model...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

    logger.info("Generating topics over time...")
    topics_over_time = pl.DataFrame()
    try:
        topics_over_time = topic_model.topics_over_time(
            docs,
            timestamps,
            global_tuning=True,
            evolution_tuning=True,
            nr_bins=20,
        )
    except Exception as e:
        logger.error(f"Could not generate topics over time: {e}")


    logger.info("Generating hierarchical topics...")
    # Gracefully handle cases where no topics are generated
    hierarchical_topics = pl.DataFrame()
    # Check if more than one topic (besides the outlier topic -1) was found
    if len(topic_model.get_topic_info()) > 1:
        try:
            hierarchical_topics = topic_model.hierarchical_topics(docs)
        except Exception as e:
            logger.error(f"Could not generate hierarchical topics: {e}")
    else:
        logger.warning("Not enough topics found to generate a hierarchy. Skipping.")

    results_df = pl.DataFrame(
        {
            "topic_id": topics,
            # "probability": probs, # This can be large, enable if needed
        }
    )

    logger.info("BERTopic pipeline complete.")
    return topic_model, results_df, topics_over_time, hierarchical_topics


if __name__ == "__main__":
    # This file is not meant to be run directly as it depends on
    # a larger data processing pipeline.
    print("BERTopic runner module. Import and use `run_bertopic` in your main pipeline.")
