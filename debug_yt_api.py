import logging
from youtube_transcript_api import YouTubeTranscriptApi

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# A video that is very likely to have a transcript
video_id = "vp_h649sZ9A" # MKBHD - The State of Foldables in 2025!

try:
    logger.info(f"Attempting to list transcripts for video ID: {video_id}")
    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    logger.info(f"Available transcripts: {[t.language_code for t in transcripts]}")

    logger.info("Attempting to fetch English transcript...")
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    logger.info("Successfully fetched transcript.")
    # print(transcript)

except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True)
