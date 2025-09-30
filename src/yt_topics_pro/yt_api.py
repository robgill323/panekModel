# src/yt_topics_pro/yt_api.py
"""
Functions for fetching data from YouTube using yt-dlp.
This module uses yt-dlp to fetch video metadata and transcripts,
as it is generally more robust than other libraries.
"""
import logging
import os
import tempfile
from typing import List, Dict, Any

import polars as pl
import yt_dlp

logger = logging.getLogger(__name__)


def parse_vtt(vtt_content: str) -> List[Dict[str, Any]]:
    """
    A simple VTT parser to extract text lines.
    Ignores timestamps and metadata.
    """
    lines = vtt_content.strip().split('\n')
    segments = []
    text_buffer = []

    # Start reading after the WEBVTT header
    in_header = True
    for line in lines:
        line = line.strip()
        if 'WEBVTT' in line:
            continue
        if '-->' in line:
            # This is a timing line, which signals the end of the previous text block.
            if text_buffer:
                segments.append({'text': ' '.join(text_buffer)})
                text_buffer = []
            continue
        if line:
            # This is a text line
            text_buffer.append(line)

    # Add the last buffered text
    if text_buffer:
        segments.append({'text': ' '.join(text_buffer)})
        
    return segments


def fetch_transcripts(video_ids: List[str]) -> pl.DataFrame:
    """
    Fetch transcripts for a list of YouTube video IDs by downloading VTT files using yt-dlp.
    """
    logger.info(f"Fetching transcripts for {len(video_ids)} videos using yt-dlp.")
    all_transcripts = []

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'outtmpl': os.path.join(tmpdir, '%(id)s.%(ext)s'),
            'quiet': True,
            'logtostderr': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for video_id in video_ids:
                try:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    info_dict = ydl.extract_info(url, download=True)
                    
                    requested_subs = info_dict.get('requested_subtitles')
                    if not requested_subs or 'en' not in requested_subs:
                        logger.warning(f"No English subtitles found or downloaded for {video_id}.")
                        continue

                    sub_info = requested_subs['en']
                    sub_filepath = sub_info.get('filepath')
                    
                    if not sub_filepath or not os.path.exists(sub_filepath):
                        logger.warning(f"Subtitle file path not found for {video_id} at {sub_filepath}.")
                        continue
                    
                    with open(sub_filepath, 'r', encoding='utf-8') as f:
                        vtt_content = f.read()
                    
                    parsed_segments = parse_vtt(vtt_content)
                    
                    for i, segment in enumerate(parsed_segments):
                        all_transcripts.append({
                            'video_id': video_id,
                            'text': segment['text'],
                            'start': float(i),  # Placeholder start time
                            'duration': 1.0,  # Placeholder duration
                            'lang': 'en'
                        })
                    logger.info(f"Successfully fetched and parsed transcript for {video_id}.")
                except Exception as e:
                    logger.error(f"Failed to process video {video_id} with yt-dlp: {e}", exc_info=False)

    if not all_transcripts:
        return pl.DataFrame({
            "video_id": [], "text": [], "start": [], "duration": [], "lang": []
        }, schema={
            "video_id": pl.Utf8, "text": pl.Utf8, "start": pl.Float64,
            "duration": pl.Float64, "lang": pl.Utf8
        })

    return pl.from_records(all_transcripts).select(["video_id", "text", "start", "duration", "lang"])


def fetch_metadata(video_ids: List[str]) -> pl.DataFrame:
    """
    Fetch basic metadata for a list of YouTube video IDs using yt-dlp.
    """
    logger.info(f"Fetching metadata for {len(video_ids)} videos using yt-dlp.")
    metadata = []
    
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for video_id in video_ids:
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                info = ydl.extract_info(url, download=False)
                metadata.append({
                    "video_id": info.get("id"),
                    "title": info.get("title"),
                    "channel_id": info.get("channel_id"),
                    "channel_name": info.get("channel"),
                    "publish_date": info.get("upload_date"), # YYYYMMDD
                    "view_count": info.get("view_count"),
                })
            except Exception as e:
                logger.error(f"Failed to fetch metadata for {video_id}: {e}")

    if not metadata:
        return pl.DataFrame({
            "video_id": [], "title": [], "channel_id": [], "channel_name": [],
            "publish_date": [], "view_count": []
        }, schema={
            "video_id": pl.Utf8, "title": pl.Utf8, "channel_id": pl.Utf8,
            "channel_name": pl.Utf8, "publish_date": pl.Utf8, "view_count": pl.Int64
        })

    df = pl.from_records(metadata)
    return df.with_columns(
        pl.col("publish_date").str.to_date(format="%Y%m%d", strict=False).alias("publish_date")
    )


def ingest_videos(video_ids: List[str]) -> Dict[str, pl.DataFrame]:
    """
    Ingest transcripts and metadata for a list of video IDs.
    """
    transcripts_df = fetch_transcripts(video_ids)
    metadata_df = fetch_metadata(video_ids)
    return {"transcripts": transcripts_df, "metadata": metadata_df}

if __name__ == '__main__':
    # Example usage
    sample_video_ids = ["iG9CE55wbtY"] 
    data = ingest_videos(sample_video_ids)
    print("--- Transcripts ---")
    print(data["transcripts"].head())
    print("\n--- Metadata ---")
    print(data["metadata"].head())
