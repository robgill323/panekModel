# src/yt_topics_pro/config.py
"""
Pydantic-based configuration for the entire application.
Settings can be loaded from environment variables or a .env file.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingSettings(BaseModel):
    """Settings for transcript chunking."""
    max_chars: int = Field(
        default=1000,
        description="Target maximum characters per chunk.",
    )
    overlap: int = Field(
        default=100,
        description="Number of characters to overlap between chunks.",
    )
    join_speaker_gaps_seconds: float = Field(
        default=5.0,
        description="Combine transcript segments if speaker gap is less than this.",
    )


class UmapSettings(BaseModel):
    """Settings for UMAP dimensionality reduction."""
    n_neighbors: int = Field(default=15, description="Number of UMAP neighbors.")
    min_dist: float = Field(default=0.1, description="Minimum distance for UMAP.")


class BertopicSettings(BaseModel):
    """Settings for BERTopic modeling."""
    min_topic_size: int = Field(default=10, description="Minimum size of a topic.")
    nr_topics: Optional[int] = Field(
        default=None, description="Number of topics to reduce to (optional)."
    )
    calculate_probabilities: bool = Field(
        default=True, description="Whether to calculate the probabilities of all topics for each document."
    )
    use_llm_labels: bool = Field(
        default=False, description="Use LLM to generate topic labels."
    )
    llm_model: str = Field(
        default="gpt-3.5-turbo", description="Model to use for LLM labeling."
    )


class SentimentSettings(BaseModel):
    """Settings for sentiment analysis models."""
    polarity: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotions: str = "SamLowe/roberta-base-go_emotions"


class EvalSettings(BaseModel):
    """Settings for topic model evaluation."""
    compute_coherence: bool = True
    coherence_measures: List[str] = ["c_v", "c_npmi", "u_mass"]


class StorageSettings(BaseModel):
    """Settings for data storage."""
    parquet_dir: str = "data/parquet"
    duckdb_path: str = "data/database.duckdb"
    faiss_dir: str = "data/faiss"
    reports_dir: str = "data/reports"
    models_dir: str = "data/models"


class DashboardSettings(BaseModel):
    """Settings for the Streamlit dashboard."""
    default_channel: str = "UCvKRFNawVcuz4b9ih_iA_zw"  # MrBeast
    show_timelines: bool = True


class AppSettings(BaseSettings):
    """Main application settings."""
    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__"
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="SentenceTransformer model for embeddings.",
    )
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    umap: UmapSettings = Field(default_factory=UmapSettings)
    bertopic: BertopicSettings = Field(default_factory=BertopicSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    evaluation: EvalSettings = Field(default_factory=EvalSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    
    # Global settings
    seed: int = 42
    gpu: bool = False # Flag to prefer GPU if available


# Instantiate settings for easy import across the project
settings = AppSettings()

if __name__ == "__main__":
    # Example of how to access settings
    print(settings.model_dump_json(indent=2))
