# debug_normalization_2.py
import polars as pl
import re
import logging

# Configure logging to see output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Function copied directly from src/yt_topics_pro/normalize.py ---
def _remove_repeated_phrases(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    # This regex finds sequences of one or more words that are immediately repeated.
    # We loop `re.sub` until no more changes are made to handle cases like 
    # "phrase phrase phrase" -> "phrase phrase" -> "phrase".
    previous_text = ""
    # The regex: \b matches a word boundary, (\w+(\s+\w+)*) captures a sequence of words,
    # \s+ matches the space between repetitions, and \1 is a backreference to the captured phrase.
    # We make it case-insensitive to catch more variations.
    pattern = re.compile(r'\b(\w+(?:\s+\w+)*)\s+\1\b', re.IGNORECASE)
    
    # Let's add logging to see what's happening inside
    logger.debug(f"Original text: '{text[:100]}...'")
    
    # The loop should handle multiple repetitions, e.g., "a b c a b c a b c"
    loop_count = 0
    while previous_text != text and loop_count < 10: # Safety break
        previous_text = text
        text = pattern.sub(r'\1', text)
        if previous_text != text:
            logger.debug(f"Loop {loop_count}: Text changed.")
        loop_count += 1
        
    logger.debug(f"Cleaned text: '{text[:100]}...'")
    return text

# --- Main execution ---
if __name__ == "__main__":
    # Load the raw data that was used in the last 'process' run
    raw_transcripts_path = "data/parquet/raw/transcripts.parquet"
    logger.info(f"Loading raw data from {raw_transcripts_path}")
    
    try:
        df = pl.read_parquet(raw_transcripts_path)
        
        # Isolate the text column
        text_series = df["text"]
        
        # Find a problematic row to test on. Let's look for the text from the user.
        # This is a simplified search.
        problem_text = None
        for text in text_series:
            if "online advertising offering search" in text:
                problem_text = text
                break
        
        if not problem_text:
            logger.warning("Could not find the specific problematic text. Using the first row as a fallback.")
            problem_text = text_series[0]

        logger.info("--- Running Test on a Problematic String ---")
        
        # Step 1: Initial Cleaning (mimicking the real pipeline)
        logger.info("Step 1: Applying initial Polars-native cleaning...")
        cleaned_series_step1 = (
            pl.Series([problem_text])
            .str.replace_all(r"<[^>]+>", " ")
            .str.replace_all(r"c\d+c?", "")
            .str.to_lowercase()
            .str.replace_all(r"[^a-z\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
        text_after_step1 = cleaned_series_step1[0]
        logger.info(f"Text after initial cleaning:\n{text_after_step1}\n")

        # Step 2: Apply the UDF
        logger.info("Step 2: Applying the _remove_repeated_phrases function...")
        text_after_step2 = _remove_repeated_phrases(text_after_step1)
        logger.info(f"Text after phrase removal function:\n{text_after_step2}\n")

        logger.info("--- Verification ---")
        if "online advertising offering search online advertising offering search" in text_after_step2:
            logger.error("FAILURE: Repeated phrases still exist after the function call.")
        else:
            logger.info("SUCCESS: Repeated phrases seem to be removed by the function.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
