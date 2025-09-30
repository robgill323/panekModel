# src/yt_topics_pro/app/plots.py
"""
Plotly functions for creating visualizations for the Streamlit dashboard.
"""
import plotly.express as px
import polars as pl


def create_umap_scatter(df: pl.DataFrame):
    """
    Creates an interactive UMAP scatter plot.

    Args:
        df: DataFrame with UMAP coordinates ('embedding_umap_x', 'embedding_umap_y'),
            topic assignments, and hover text.

    Returns:
        A Plotly Figure object.
    """
    fig = px.scatter(
        df.to_pandas(),  # Plotly Express works best with Pandas
        x="embedding_umap_x",
        y="embedding_umap_y",
        color="topic",
        hover_data=["video_id", "start", "text"],
        title="UMAP Projection of Video Chunks by Topic",
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig.update_layout(
        legend_title_text='Topic',
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
    )
    fig.update_traces(
        textposition='top center',
        hovertemplate=(
            "<b>Video:</b> %{customdata[0]}<br>"
            "<b>Time:</b> %{customdata[1]:.2f}s<br>"
            "<b>Text:</b> %{customdata[2]|<br>s}" # Wrap text
        )
    )
    return fig


def create_topics_over_time_chart(df: pl.DataFrame):
    """
    Creates a line chart showing topic frequency over time.

    Args:
        df: The topics_over_time DataFrame from BERTopic.

    Returns:
        A Plotly Figure object.
    """
    fig = px.line(
        df.to_pandas(),
        x="Timestamp",
        y="Frequency",
        color="Topic",
        title="Topic Frequency Over Time",
    )
    return fig


def create_sentiment_bars(df: pl.DataFrame):
    """
    Creates bar charts for sentiment per topic.

    Args:
        df: DataFrame with aggregated sentiment scores per topic.

    Returns:
        A Plotly Figure object.
    """
    # This requires melting the data into a long format
    id_vars = ["topic"]
    value_vars = [c for c in df.columns if "polarity_" in c or "emotion_" in c]
    
    if not value_vars:
        return None

    long_df = df.melt(id_vars=id_vars, value_vars=value_vars,
                      variable_name="sentiment", value_name="score")

    fig = px.bar(
        long_df.to_pandas(),
        x="topic",
        y="score",
        color="sentiment",
        barmode="group",
        title="Average Sentiment Scores by Topic",
    )
    return fig
