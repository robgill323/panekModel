# src/yt_topics_pro/app/dashboard.py
"""
Streamlit dashboard for exploring YouTube topic modeling results.
"""
import streamlit as st
import polars as pl
import logging

from yt_topics_pro.config import settings
from yt_topics_pro.storage.duck import query_duckdb, register_parquet_views
from . import plots

logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="YT Topics Pro")


def load_data():
    """Loads all necessary data from DuckDB/Parquet."""
    try:
        register_parquet_views()
        chunks_df = query_duckdb("SELECT * FROM processed_chunks").pl()
        # topic_sentiment_df = query_duckdb("SELECT * FROM processed_topic_sentiment").pl()
        # Add other tables as needed
        return chunks_df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        logger.error(f"Dashboard data loading error: {e}")
        return None


def main():
    """Main function to render the Streamlit dashboard."""
    st.title("YouTube Topics Explorer")

    # Load data
    data = load_data()
    if data is None:
        st.warning("No data found. Please run the processing pipeline first.")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    
    # TODO: Add filters for channel, date range, etc.
    # Example:
    # unique_channels = data["channel_name"].unique().to_list()
    # selected_channel = st.sidebar.selectbox("Channel", unique_channels)
    
    # Filter data based on selections
    # filtered_data = data.filter(pl.col("channel_name") == selected_channel)
    filtered_data = data # No filter for now

    # --- Main Content ---
    st.header("Topic & Sentiment Analysis")

    tab1, tab2, tab3 = st.tabs(["UMAP Scatter", "Topic Details", "Video View"])

    with tab1:
        st.subheader("UMAP Scatterplot of Text Chunks")
        if "embedding_umap_x" in filtered_data.columns:
            fig = plots.create_umap_scatter(filtered_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("UMAP coordinates not found in the data.")

    with tab2:
        st.subheader("Explore Individual Topics")
        # TODO: Display topic cards with top words, sentiment, etc.
        st.info("Topic detail view is not yet implemented.")

    with tab3:
        st.subheader("Per-Video Analysis")
        # TODO: Show topic mix and sentiment timeline for selected videos.
        st.info("Video-level view is not yet implemented.")


if __name__ == "__main__":
    # To run this:
    # 1. Make sure you have run the full `ingest` and `process` pipeline.
    # 2. Run from your terminal: `streamlit run src/yt_topics_pro/app/dashboard.py`
    logging.basicConfig(level=logging.INFO)
    main()
