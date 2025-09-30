# yt-topics-pro

This project is an end-to-end pipeline for topic modeling and sentiment analysis of YouTube video transcripts.

## Features

-   **Ingestion**: Fetches video transcripts and metadata from YouTube channels or a list of video URLs.
-   **Processing**: Chunks text, generates sentence embeddings, performs topic modeling (BERTopic), and computes sentiment/emotions.
-   **Storage**: Persists all data efficiently using Parquet, DuckDB, and FAISS.
-   **Evaluation**: Calculates topic quality metrics like coherence and diversity.
-   **Dashboard**: Provides an interactive Streamlit dashboard to explore the results.

## Quickstart

1.  **Install dependencies:**

    This project uses Poetry for dependency management.

    ```bash
    pip install poetry
    poetry install
    ```

2.  **Create a video list:**

    Create a file named `videos.txt` and add some YouTube video URLs, one per line.

    ```
    https://www.youtube.com/watch?v=...
    https://www.youtube.com/watch?v=...
    ```

3.  **Run the pipeline:**

    The CLI is managed by Typer. Use `yt-topics-pro --help` to see all commands.

    ```bash
    # Ingest the videos
    poetry run yt-topics-pro ingest --videos videos.txt

    # Process the data (chunk, embed, model, sentiment)
    poetry run yt-topics-pro process

    # (Optional) Evaluate the topic model
    poetry run yt-topics-pro evaluate

    # Launch the interactive dashboard
    poetry run streamlit run src/yt_topics_pro/app/dashboard.py
    ```

## Project Structure

```
yt-topics-pro/
├── data/                  # Output data (Parquet, FAISS, reports)
├── notebooks/             # Example notebooks
├── src/
│   └── yt_topics_pro/
│       ├── app/           # Streamlit dashboard and plots
│       ├── sentiment/     # Sentiment and emotion analysis modules
│       ├── storage/       # Data persistence (Parquet, DuckDB, FAISS)
│       ├── topics/        # Topic modeling and evaluation
│       ├── __init__.py
│       ├── chunking.py    # Transcript chunking logic
│       ├── cli.py         # Typer CLI application
│       ├── config.py      # Pydantic settings
│       ├── embed.py       # Embedding model loader
│       └── yt_api.py      # YouTube data fetching
├── pyproject.toml         # Project dependencies and metadata
└── README.md
```

## Configuration

Application settings are managed in `src/yt_topics_pro/config.py` using Pydantic. You can override settings via a `.env` file in the project root.

Example `.env` file:

```
EMBEDDING_MODEL="bge-large-en-v1.5"
BERTOTPIC__USE_LLM_LABELS=True
# Requires OPENAI_API_KEY to be set in the environment
```

*(Placeholder for screenshots of the dashboard)*
