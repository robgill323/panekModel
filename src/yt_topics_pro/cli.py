# src/yt_topics_pro/cli.py
"""
Typer-based CLI for the yt-topics-pro pipeline.
"""
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

import typer
import polars as pl
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

from yt_topics_pro import config, yt_api, chunking, embed, normalize
from yt_topics_pro.storage import tables as storage_tables
from yt_topics_pro.topics import bertopic_runner, evaluate
from yt_topics_pro.sentiment import polarity, emotions, aggregate

# --- Setup ---
app = typer.Typer(
    name="yt-topics-pro",
    help="A CLI for end-to-end topic modeling of YouTube videos.",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
# Enable debug logging for the transcript API to diagnose issues
logging.getLogger("youtube_transcript_api").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


# --- Helper Functions ---
def _load_video_ids_from_file(filepath: Path) -> List[str]:
    """Loads video IDs/URLs from a file, one per line, and extracts the video ID."""
    if not filepath.is_file():
        logger.error(f"Video file not found: {filepath}")
        raise typer.Exit(code=1)
    
    video_ids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # If it's a full URL, parse it
            if "youtube.com" in line or "youtu.be" in line:
                parsed_url = urlparse(line)
                if parsed_url.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
                    video_id = parse_qs(parsed_url.query).get('v')
                    if video_id:
                        video_ids.append(video_id[0])
                    else:
                        logger.warning(f"Could not parse video ID from URL: {line}")
                # youtu.be shortlinks
                elif parsed_url.hostname == 'youtu.be':
                    video_ids.append(parsed_url.path[1:])
            else:
                # Assume it's already a video ID
                video_ids.append(line)
    return video_ids


# --- CLI Commands ---

@app.command()
def ingest(
    videos_file: Optional[Path] = typer.Option(
        None,
        "--videos",
        "-v",
        help="Path to a text file containing YouTube video URLs or IDs, one per line.",
    ),
    channel_id: Optional[str] = typer.Option(
        None, "--channel", "-c", help="YouTube channel ID to ingest from."
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit the number of videos to process."),
):
    """
    1) Fetch transcripts and metadata for YouTube videos.
    """
    if not videos_file and not channel_id:
        logger.error("Please provide either a --videos file or a --channel ID.")
        raise typer.Exit(code=1)

    logger.info("🚀 Starting ingestion process...")

    if videos_file:
        video_ids = _load_video_ids_from_file(videos_file)
    else:
        # TODO: Implement channel fetching logic
        logger.warning("Channel fetching is not yet implemented.")
        video_ids = []

    if limit:
        video_ids = video_ids[:limit]

    if not video_ids:
        logger.warning("No video IDs to process.")
        raise typer.Exit()

    # Fetch data
    raw_data = yt_api.ingest_videos(video_ids)

    # Save to Parquet
    storage_tables.save_to_parquet(raw_data, "raw")
    logger.info("✅ Ingestion complete. Raw data saved to Parquet.")


@app.command()
def process(
    use_llm_labels: bool = typer.Option(
        False, "--use-llm-labels", help="Use an LLM for topic labeling (requires API key)."
    ),
    num_topics: Optional[int] = typer.Option(
        None, "--num-topics", help="Reduce the number of topics to a fixed number."
    ),
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU acceleration for models."),
    sample: Optional[int] = typer.Option(None, "--sample", help="Process a random sample of chunks."),
):
    """
    2) Chunk, embed, model topics, and compute sentiment.
    """
    logger.info("🚀 Starting processing pipeline...")
    config.settings.gpu = gpu
    config.settings.bertopic.use_llm_labels = use_llm_labels
    config.settings.bertopic.nr_topics = num_topics

    # Load raw data
    raw_tables = storage_tables.load_from_parquet("raw")
    if "transcripts" not in raw_tables or "metadata" not in raw_tables:
        logger.error("No raw transcript or metadata data found. Please run `ingest` first.")
        raise typer.Exit(code=1)

    transcripts_df = raw_tables["transcripts"].collect()
    metadata_df = raw_tables["metadata"].collect()

    # Normalize Text
    normalized_transcripts_df = normalize.normalize_transcripts(transcripts_df)

    # Chunking
    chunks_df = chunking.chunk_transcript(normalized_transcripts_df)

    # Join with metadata to get titles
    chunks_df = chunks_df.explode("video_id").join(
        metadata_df.select(["video_id", "title"]), on="video_id", how="left"
    )
    
    if sample:
        chunks_df = chunks_df.sample(n=sample, seed=config.settings.seed)

    # Embedding
    docs = chunks_df["normalized_text"].to_list()
    embeddings = embed.embed_texts(docs)
    
    # Add UMAP embeddings for plotting
    # Note: This assumes the embedding model is a SentenceTransformer pipeline with a 'umap' step
    # This might fail if the model architecture is different.
    try:
        umap_model = embed.get_embedding_model().named_steps['umap']
        umap_embeddings = umap_model.transform(embeddings.cpu().numpy())
        chunks_df = chunks_df.with_columns([
            pl.Series("embedding_umap_x", umap_embeddings[:, 0]),
            pl.Series("embedding_umap_y", umap_embeddings[:, 1]),
        ])
    except (KeyError, AttributeError):
        logger.warning("Could not get UMAP model from embedding pipeline. Skipping UMAP coordinates.")


    # Topic Modeling (BERTopic)
    timestamps = chunks_df["start_time"].to_list()
    model, topics_df, tot_df, hier_df = bertopic_runner.run_bertopic(
        docs, embeddings.cpu().numpy(), timestamps
    )

    # Save BERTopic model and its components
    models_dir = Path(config.settings.storage.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "bertopic_model"
    embeddings_path = models_dir / "embeddings.npy"
    probabilities_path = models_dir / "probabilities.npy"
    
    model.save(str(model_path), serialization="safetensors")
    np.save(embeddings_path, embeddings.cpu().numpy())
    if model.probabilities_ is not None:
        np.save(probabilities_path, model.probabilities_)
    
    logger.info(f"BERTopic model saved to {model_path}")
    logger.info(f"Embeddings saved to {embeddings_path}")
    logger.info(f"Probabilities saved to {probabilities_path}")

    # Sentiment Analysis
    polarity_df = polarity.predict_polarity(docs)
    emotions_df = emotions.predict_emotions(docs)

    # Combine all results
    processed_df = pl.concat([chunks_df, topics_df, polarity_df, emotions_df], how="horizontal")
    
    # Save processed data
    storage_tables.save_to_parquet({"chunks": processed_df}, "processed")
    
    logger.info("✅ Processing complete. Enriched data saved.")


@app.command()
def evaluate():
    """
    3) Evaluate topic model quality (coherence and diversity).
    """
    logger.info("🚀 Starting evaluation...")
    from .topics import evaluate as topic_evaluator
    topic_evaluator.run_evaluation()
    logger.info("✅ Evaluation complete.")


@app.command()
def dashboard():
    """
    4) Launch the Streamlit dashboard.
    """
    logger.info("🚀 Launching Streamlit dashboard...")
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    if not dashboard_path.exists():
        logger.error(f"Dashboard script not found at {dashboard_path}")
        raise typer.Exit(1)
        
    import subprocess
    try:
        subprocess.run(["streamlit", "run", str(dashboard_path)], check=True)
    except FileNotFoundError:
        logger.error("`streamlit` command not found. Please install it with `pip install streamlit`.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Streamlit dashboard: {e}")


if __name__ == "__main__":
    app()
