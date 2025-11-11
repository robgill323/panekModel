# src/yt_topics_pro/dashboard/app.py
"""
Streamlit dashboard for visualizing topic modeling results.
"""
import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from bertopic import BERTopic
import plotly.graph_objects as go

from yt_topics_pro.config import settings
from yt_topics_pro.storage import tables
from yt_topics_pro.embed import get_embedding_model

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="YouTube Topics Pro",
    page_icon="📊",
)

# --- Caching Functions ---
@st.cache_data(show_spinner=False)
def load_data():
    """Loads all necessary data from disk and caches it."""
    processed_data = tables.load_from_parquet("processed")
    chunks_df = processed_data.get("chunks", pl.DataFrame()).collect()
    
    models_dir = Path(settings.storage.models_dir)
    model_path = models_dir / "bertopic_model"
    embeddings_path = models_dir / "embeddings.npy"
    probabilities_path = models_dir / "probabilities.npy"

    topic_model, embeddings = None, None
    if model_path.exists() and embeddings_path.exists():
        embedding_model = get_embedding_model()
        topic_model = BERTopic.load(str(model_path), embedding_model=embedding_model)
        embeddings = np.load(embeddings_path)
        if probabilities_path.exists():
            topic_model.probabilities_ = np.load(probabilities_path)
    
    # Add a unique index for filtering
    if not chunks_df.is_empty():
        chunks_df = chunks_df.with_row_index(name="index")
        
    return chunks_df, topic_model, embeddings

@st.cache_data(show_spinner=False)
def get_topic_info_df(_topic_model):
    """Caches the topic info dataframe."""
    return _topic_model.get_topic_info()

@st.cache_data(show_spinner=False)
def generate_intertopic_map(_topic_model):
    """Caches the intertopic distance map."""
    return _topic_model.visualize_topics()

@st.cache_data(show_spinner=False)
def generate_hierarchy_chart(_topic_model):
    """Caches the topic hierarchy chart."""
    try:
        return _topic_model.visualize_hierarchy()
    except Exception:
        return None

# --- Main App ---
def run_dashboard():
    """Main function to run the Streamlit dashboard."""
    st.title("📊 YouTube Topics Pro Dashboard")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Controls & Info")
        st.info(
            "This dashboard visualizes the results of the YouTube Topics Pro pipeline. "
            "Use the tabs to navigate through different analysis views."
        )
        
        st.header("📖 How to Use")
        st.markdown(
            """
            - **Overview**: Get a high-level summary and search the raw data.
            - **Topic Deep Dive**: Select a topic to see its keywords and most relevant documents.
            - **Advanced Visuals**: Explore relationships between topics.
            - **Evaluation**: Review the quantitative quality scores for each topic.
            """
        )

        st.header("💾 Data Source")
        processed_dir = Path(settings.storage.parquet_dir) / "processed"
        if processed_dir.exists():
            st.success(f"Data loaded from `{processed_dir}`")
        else:
            st.error("Processed data not found. Run the `process` command.")
            return

    # --- Load Data ---
    with st.spinner("Loading data and models..."):
        chunks_df, topic_model, embeddings = load_data()

    if chunks_df.is_empty() or topic_model is None:
        st.error("Processed data or topic model not found. Please run the `process` command.")
        return
        
    topic_info_df = get_topic_info_df(topic_model)

    # --- Main Content Tabs ---
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload & Process",
        "📊 Overview & Data Explorer", 
        "🧠 Topic Deep Dive", 
        "🎬 Video Deep Dive",
        "📈 Advanced Visuals", 
        "📝 Model Evaluation",
        "😊 Sentiment Analysis"
    ])

    # --- Tab 0: Upload & Process ---
    with tab0:
        st.header("📤 Upload YouTube URLs & Process")
        st.markdown("""
        Upload a CSV file containing YouTube URLs to ingest transcripts and run topic modeling.
        
        **CSV Format Requirements:**
        - Must have a column named `url` containing YouTube video URLs
        - Example: `https://www.youtube.com/watch?v=VIDEO_ID`
        - One URL per row
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                help="Upload a CSV with a 'url' column containing YouTube video URLs"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded CSV
                    urls_df = pd.read_csv(uploaded_file)
                    
                    # Validate the CSV has a 'url' column
                    if 'url' not in urls_df.columns:
                        st.error("❌ CSV must contain a 'url' column!")
                    else:
                        st.success(f"✅ Loaded {len(urls_df)} URLs from CSV")
                        
                        # Show preview of URLs
                        with st.expander("Preview URLs"):
                            st.dataframe(urls_df.head(10), use_container_width=True)
                        
                        # Save to temporary text file (one URL per line)
                        temp_urls_path = Path(settings.storage.parquet_dir) / "temp_urls.txt"
                        temp_urls_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(temp_urls_path, 'w') as f:
                            for url in urls_df['url']:
                                f.write(f"{url}\n")
                        
                        st.info(f"📁 URLs saved to: `{temp_urls_path}`")
                        
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
        
        with col2:
            st.subheader("⚙️ Processing Options")
            
            use_llm_labels = st.checkbox(
                "Use LLM for topic labels",
                value=False,
                help="Use GPT to generate descriptive topic names (requires API key)"
            )
            
            num_topics = st.number_input(
                "Number of topics",
                min_value=0,
                max_value=500,
                value=0,
                help="Leave at 0 for automatic topic discovery, or set a specific number to reduce topics"
            )
            
            if num_topics == 0:
                num_topics = None
        
        st.markdown("---")
        
        # Processing buttons
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔽 1. Ingest Transcripts", use_container_width=True, type="primary"):
                if uploaded_file is None:
                    st.warning("⚠️ Please upload a CSV file first!")
                else:
                    with st.spinner("Ingesting transcripts... This may take several minutes."):
                        try:
                            import subprocess
                            import sys
                            
                            # Run the ingest command
                            result = subprocess.run(
                                [sys.executable, "-m", "yt_topics_pro.cli", "ingest", "--videos", str(temp_urls_path)],
                                capture_output=True,
                                text=True,
                                cwd=str(Path(__file__).parent.parent.parent.parent)
                            )
                            
                            if result.returncode == 0:
                                st.success("✅ Transcripts ingested successfully!")
                                st.code(result.stdout, language="text")
                            else:
                                st.error(f"❌ Ingestion failed with error code {result.returncode}")
                                st.code(result.stderr, language="text")
                        except Exception as e:
                            st.error(f"❌ Error running ingestion: {str(e)}")
        
        with col_btn2:
            if st.button("⚙️ 2. Process & Model Topics", use_container_width=True, type="primary"):
                with st.spinner("Processing data and modeling topics... This will take several minutes."):
                    try:
                        import subprocess
                        import sys
                        
                        # Build command arguments
                        cmd = [sys.executable, "-m", "yt_topics_pro.cli", "process"]
                        if use_llm_labels:
                            cmd.append("--use-llm-labels")
                        if num_topics is not None:
                            cmd.extend(["--num-topics", str(num_topics)])
                        
                        # Run the process command
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            cwd=str(Path(__file__).parent.parent.parent.parent)
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Processing complete! Refresh the page to see results.")
                            st.code(result.stdout[-2000:], language="text")  # Show last 2000 chars
                            
                            # Clear the cache so new data loads
                            st.cache_data.clear()
                        else:
                            st.error(f"❌ Processing failed with error code {result.returncode}")
                            st.code(result.stderr, language="text")
                    except Exception as e:
                        st.error(f"❌ Error running processing: {str(e)}")
        
        with col_btn3:
            if st.button("🔄 3. Refresh Dashboard", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Instructions
        st.markdown("---")
        st.subheader("📖 How to Use")
        st.markdown("""
        1. **Upload CSV**: Upload a CSV file with YouTube URLs in a column named `url`
        2. **Ingest**: Click "1. Ingest Transcripts" to download and extract transcripts
        3. **Process**: Click "2. Process & Model Topics" to run topic modeling
        4. **Refresh**: Click "3. Refresh Dashboard" to load the new results in other tabs
        
        **Note:** Processing can take 10-60 minutes depending on the number of videos.
        """)

    # --- Tab 1: Overview & Data Explorer ---
    with tab1:
        st.header("Project Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks Processed", f"{chunks_df.shape[0]:,}")
        col2.metric("Total Videos Analyzed", f"{chunks_df['video_id'].n_unique():,}")
        col3.metric("Topics Discovered", f"{topic_info_df.shape[0] - 1}")

        st.header("Data Explorer")
        st.markdown("Search and filter the processed text chunks used for topic modeling.")
        
        # Filtering and Searching
        filtered_df = chunks_df
        video_list = filtered_df["video_id"].unique().to_list()
        selected_videos = st.multiselect("Filter by Video ID:", options=video_list)
        search_query = st.text_input("Search text chunks:", placeholder="e.g., 'machine learning'")

        if selected_videos:
            filtered_df = filtered_df.filter(pl.col("video_id").is_in(selected_videos))
        if search_query:
            filtered_df = filtered_df.filter(pl.col("normalized_text").str.contains(search_query, literal=False))
        
        # Prepare topic names for joining
        topic_info_pl_df = pl.from_pandas(topic_info_df).rename({"Topic": "topic_id"})
        
        # Create display dataframe with URLs and topic names
        display_df = filtered_df.with_columns(
            pl.format("https://www.youtube.com/watch?v={}", pl.col("video_id")).alias("video_url")
        ).join(
            topic_info_pl_df.select(["topic_id", "Name"]), on="topic_id", how="left"
        ).select([
            "video_url",
            "title", 
            "Name",
            "normalized_text",
            "sentiment"
        ]).rename({
            "video_url": "Video URL",
            "title": "Video Title",
            "Name": "Topic",
            "normalized_text": "Text",
            "sentiment": "Sentiment"
        })
        
        st.dataframe(display_df, use_container_width=True)

    # --- Tab 2: Topic Deep Dive ---
    with tab2:
        st.header("Explore Individual Topics")
        
        with st.container(border=True):
            topic_options = {
                row["Topic"]: f"Topic {row['Topic']}: {row['Name']}"
                for _, row in topic_info_df.iterrows()
                if row["Topic"] != -1
            }
            selected_topic_id = st.selectbox(
                "**Choose a topic to inspect:**",
                options=list(topic_options.keys()),
                format_func=lambda x: topic_options.get(x, "Select a Topic"),
            )

        st.write("") # Add some vertical space

        if selected_topic_id is not None:
            col1, col2 = st.columns([1, 2], gap="large")
            
            with col1:
                with st.container(border=True):
                    topic_name = topic_options[selected_topic_id]
                    st.subheader(f"📊 Top Keywords for: {topic_name}")
                    st.markdown("The most important words for the selected topic, based on c-TF-IDF score.")

                    # --- Custom Barchart Generation ---
                    # Get the data directly from the model
                    topic_data = topic_model.get_topic(selected_topic_id)
                    
                    if topic_data:
                        words = [item[0] for item in topic_data][:10]
                        scores = [item[1] for item in topic_data][:10]
                        
                        # Create a new Plotly figure from scratch for full control
                        fig = go.Figure(go.Bar(
                            x=scores,
                            y=words,
                            orientation='h',
                            marker_color='#1c83e1',
                            marker_line_width=0
                        ))
                        
                        # Apply the sleek layout
                        fig.update_layout(
                            title_text="", # Explicitly remove title
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="c-TF-IDF Score",
                            yaxis=dict(autorange="reversed"), # Show top score at the top
                            yaxis_title=None,
                            font=dict(family="sans serif", size=12, color="#31333F")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not retrieve data for this topic.")

            with col2:
                with st.container(border=True):
                    st.subheader("📄 Representative Documents")
                    st.markdown("Text chunks that are most representative of this topic.")
                    docs = topic_model.get_representative_docs(selected_topic_id)
                    if docs:
                        for doc in docs:
                            st.info(f"_{doc}_")
                    else:
                        st.warning("No single representative document found. Showing top 5 most relevant instead.")
                        probs = topic_model.probabilities_
                        if probs is not None:
                            topic_list = list(topic_info_df["Topic"])
                            topic_idx = topic_list.index(selected_topic_id)
                            top_5_indices = np.argsort(probs[:, topic_idx])[-5:][::-1]
                            top_docs = chunks_df.filter(pl.col("index").is_in(top_5_indices))["normalized_text"]
                            for doc in top_docs:
                                st.info(f"_{doc}_")

    # --- Tab 3: Video Deep Dive ---
    with tab3:
        st.header("🎬 Video Deep Dive Analysis")
        st.markdown("Select a video to analyze its topic distribution, sentiment, and content in detail.")
        
        # Get unique videos with titles
        video_options_df = chunks_df.select(["video_id", "title"]).unique().sort("title")
        video_dict = {row["video_id"]: row["title"] for row in video_options_df.iter_rows(named=True)}
        
        selected_video = st.selectbox(
            "**Choose a video to analyze:**",
            options=list(video_dict.keys()),
            format_func=lambda x: f"{video_dict[x][:80]}..." if len(video_dict[x]) > 80 else video_dict[x]
        )
        
        if selected_video:
            # Filter chunks for this video
            video_chunks = chunks_df.filter(pl.col("video_id") == selected_video)
            
            # Get video metadata
            video_title = video_dict[selected_video]
            video_url = f"https://www.youtube.com/watch?v={selected_video}"
            
            # Display video info header
            st.subheader(f"📹 {video_title}")
            st.markdown(f"🔗 [Watch on YouTube]({video_url})")
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                # Embedded YouTube player
                with st.container(border=True):
                    st.markdown("### Video Preview")
                    embed_url = f"https://www.youtube.com/embed/{selected_video}"
                    st.markdown(
                        f'<iframe width="100%" height="315" src="{embed_url}" '
                        f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
                        f'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
                        unsafe_allow_html=True
                    )
                
                # Video statistics
                with st.container(border=True):
                    st.markdown("### Video Statistics")
                    stats_col1, stats_col2 = st.columns(2)
                    stats_col1.metric("Total Chunks", f"{video_chunks.shape[0]}")
                    
                    avg_sentiment = video_chunks["sentiment"].mean()
                    sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                    stats_col2.metric("Avg Sentiment", f"{sentiment_label} ({avg_sentiment:.2f})")
            
            with col2:
                # Topic distribution pie chart
                with st.container(border=True):
                    st.markdown("### Topic Distribution")
                    
                    # Calculate topic counts
                    topic_counts = video_chunks.filter(pl.col("topic_id") != -1).group_by("topic_id").len()
                    
                    if not topic_counts.is_empty():
                        # Join with topic names
                        topic_info_pl_df = pl.from_pandas(topic_info_df).rename({"Topic": "topic_id"})
                        topic_dist = topic_counts.join(
                            topic_info_pl_df.select(["topic_id", "Name"]), 
                            on="topic_id", 
                            how="left"
                        ).sort("len", descending=True)
                        
                        # Get dominant topic
                        dominant_topic = topic_dist.row(0, named=True)
                        st.info(f"**Dominant Topic:** {dominant_topic['Name']} ({dominant_topic['len']} chunks, {dominant_topic['len']/video_chunks.shape[0]*100:.1f}%)")
                        
                        # Create pie chart
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=topic_dist["Name"].to_list(),
                            values=topic_dist["len"].to_list(),
                            hole=0.3,
                            marker=dict(
                                colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                       '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
                            )
                        )])
                        
                        fig_pie.update_layout(
                            showlegend=True,
                            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=350
                        )
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("No topics assigned to chunks in this video.")
            
            # Sentiment timeline
            st.markdown("---")
            with st.container(border=True):
                st.markdown("### Sentiment Over Time")
                st.markdown("Track how sentiment changes throughout the video.")
                
                # Sort by chunk_id or start_time if available
                if "start_time" in video_chunks.columns:
                    timeline_data = video_chunks.sort("start_time")
                elif "chunk_id" in video_chunks.columns:
                    timeline_data = video_chunks.sort("chunk_id")
                else:
                    timeline_data = video_chunks
                
                # Create line chart
                fig_sentiment = go.Figure()
                
                fig_sentiment.add_trace(go.Scatter(
                    x=list(range(len(timeline_data))),
                    y=timeline_data["sentiment"].to_list(),
                    mode='lines+markers',
                    name='Sentiment',
                    line=dict(color='#1c83e1', width=2),
                    marker=dict(size=6),
                    fill='tozeroy',
                    fillcolor='rgba(28, 131, 225, 0.1)'
                ))
                
                # Add zero line
                fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                fig_sentiment.update_layout(
                    xaxis_title="Chunk Sequence",
                    yaxis_title="Sentiment Score",
                    yaxis=dict(range=[-1, 1]),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300
                )
                
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # Chunk details table
            st.markdown("---")
            with st.container(border=True):
                st.markdown("### Chunk Details")
                st.markdown("All text chunks from this video with their topics and sentiment.")
                
                # Prepare display dataframe
                topic_info_pl_df = pl.from_pandas(topic_info_df).rename({"Topic": "topic_id"})
                
                chunk_display = video_chunks.join(
                    topic_info_pl_df.select(["topic_id", "Name"]), 
                    on="topic_id", 
                    how="left"
                ).select([
                    "chunk_id" if "chunk_id" in video_chunks.columns else pl.lit(None).alias("chunk_id"),
                    "Name",
                    "normalized_text",
                    "sentiment"
                ]).rename({
                    "chunk_id": "Chunk",
                    "Name": "Topic",
                    "normalized_text": "Text",
                    "sentiment": "Sentiment"
                })
                
                st.dataframe(chunk_display, use_container_width=True, height=400)

    # --- Tab 4: Advanced Visualizations ---
    with tab4:
        st.header("Intertopic Distance Map")
        st.markdown("Visualize topics as circles, where size indicates prevalence and distance indicates similarity.")
        with st.spinner("Generating Intertopic Map..."):
            fig_topics = generate_intertopic_map(topic_model)
            st.plotly_chart(fig_topics, use_container_width=True)

        st.header("Topic Hierarchy")
        st.markdown(
            """
            The Topic Hierarchy visualization, also known as a dendrogram, illustrates how topics can be grouped together into broader themes.

            *   **What you're seeing:** A tree-like diagram where each leaf at the bottom represents one of the discovered topics.
            *   **How to interpret it:** As you move up the tree, similar topics are merged together. The height of the horizontal line connecting two topics (or topic clusters) indicates how dissimilar they are.
            *   **Why it's useful:** This view helps you understand the relationships between topics. Topics that merge low down on the tree are very similar, while topics that merge higher up are less related. It can reveal overarching themes in your data.
            """
        )
        with st.spinner("Generating Hierarchy Chart..."):
            fig_hierarchy = generate_hierarchy_chart(topic_model)
            if fig_hierarchy:
                st.plotly_chart(fig_hierarchy, use_container_width=True)
            else:
                st.warning("Could not generate hierarchy visualization. This can happen with too few topics.")

    # --- Tab 5: Model Evaluation ---
    with tab5:
        st.header("Topic Model Evaluation Metrics")
        st.markdown("This table shows quantitative metrics for each topic, helping to assess their quality.")
        evaluation_path = Path(settings.storage.reports_dir) / "topic_model_evaluation.csv"
        if not evaluation_path.exists():
            st.warning("No evaluation report found. Please run the `evaluate` command first.")
        else:
            eval_df = pd.read_csv(evaluation_path)
            st.markdown("Higher coherence scores (like **c_v** and **c_npmi**) generally indicate more interpretable topics.")
            
            # Check if columns for styling exist
            style_cols = ['c_v', 'c_npmi', 'u_mass']
            available_style_cols = [col for col in style_cols if col in eval_df.columns]
            
            if available_style_cols:
                st.dataframe(
                    eval_df.style.background_gradient(
                        cmap='Greens',
                        subset=available_style_cols
                    ),
                    use_container_width=True
                )
            else:
                st.dataframe(eval_df, use_container_width=True)

    # --- Tab 6: Sentiment Analysis ---
    with tab6:
        st.header("Sentiment Analysis by Video and Topic")
        st.markdown(
            """
            This section breaks down the average sentiment scores.
            - **Per Video**: Shows the dominant topic and average sentiment for each video.
            - **Per Topic**: Shows the overall average sentiment for all content within a topic.
            """
        )

        if chunks_df.is_empty() or "sentiment" not in chunks_df.columns:
            st.warning("Sentiment data not available. Please ensure sentiment analysis was run during processing.")
        else:
            # --- Video-level Analysis ---
            st.subheader("Average Sentiment & Dominant Topic per Video")

            # Calculate average sentiment per video
            video_sentiment = chunks_df.group_by("video_id").agg(
                pl.mean("sentiment").alias("avg_sentiment"),
                pl.first("title").alias("title") # Get video title
            )

            # Determine dominant topic for each video
            video_topic_counts = chunks_df.filter(pl.col("topic_id") != -1).group_by(["video_id", "topic_id"]).len()
            
            # Handle cases where a video might not have any assigned topics
            if not video_topic_counts.is_empty():
                video_dominant_topic = (
                    video_topic_counts
                    .sort("len", descending=True)
                    .group_by("video_id", maintain_order=True)
                    .first()
                )

                # Join sentiment and topic info
                video_analysis_df = video_sentiment.join(
                    video_dominant_topic, on="video_id", how="left"
                )

                # Prepare topic names for joining
                topic_info_pl_df = pl.from_pandas(topic_info_df).rename({"Topic": "topic_id"})

                # Join with topic names
                video_analysis_df = video_analysis_df.join(
                    topic_info_pl_df.select(["topic_id", "Name"]), on="topic_id", how="left"
                )
                
                # Add URL column
                video_analysis_df = video_analysis_df.with_columns(
                    pl.format("https://www.youtube.com/watch?v={}", pl.col("video_id")).alias("URL")
                ).select(
                    ["title", "URL", "Name", "avg_sentiment"]
                ).rename({"Name": "Dominant Topic", "avg_sentiment": "Avg. Sentiment"})
                
                st.dataframe(video_analysis_df, use_container_width=True)
            else:
                st.info("No topic assignments found to determine dominant topics for videos.")


            # --- Topic-level Analysis ---
            st.subheader("Average Sentiment per Topic")
            topic_sentiment = chunks_df.filter(pl.col("topic_id") != -1).group_by("topic_id").agg(
                pl.mean("sentiment").alias("avg_topic_sentiment")
            )
            
            topic_info_pl_df = pl.from_pandas(topic_info_df).rename({"Topic": "topic_id"})
            
            topic_sentiment = topic_sentiment.join(
                topic_info_pl_df.select(["topic_id", "Name"]), on="topic_id", how="inner"
            ).sort("avg_topic_sentiment", descending=True)
            
            st.dataframe(
                topic_sentiment.select(["Name", "avg_topic_sentiment"]).rename(
                    {"Name": "Topic", "avg_topic_sentiment": "Avg. Sentiment"}
                ), 
                use_container_width=True
            )

if __name__ == "__main__":
    run_dashboard()
