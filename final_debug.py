# final_debug.py
import polars as pl
import re
import logging
from pathlib import Path

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DEBUG_DIR = Path("data/debug")
DEBUG_DIR.mkdir(exist_ok=True)

# --- Functions from normalize.py ---
def _remove_repeated_phrases(text: str) -> str:
    if not isinstance(text, str):
        return text
    pattern = re.compile(r'\b(\w+(?:\s+\w+)*)\s+\1\b', re.IGNORECASE)
    previous_text = ""
    while previous_text != text:
        previous_text = text
        text = pattern.sub(r'\1', text)
    return text

DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
    "yourselves"
}

# --- Main Debugging Logic ---
if __name__ == "__main__":
    raw_transcripts_path = "data/parquet/raw/transcripts.parquet"
    logger.info(f"Loading raw data from {raw_transcripts_path}")
    
    try:
        df = pl.read_parquet(raw_transcripts_path)
        df.write_csv(DEBUG_DIR / "00_original.csv")
        logger.info("Saved 00_original.csv")

        # --- Step 1: Phrase Removal ---
        logger.info("Running Step 1: _remove_repeated_phrases")
        df_step1 = df.with_columns(
            pl.col("text").map_elements(_remove_repeated_phrases, return_dtype=pl.String).alias("text")
        )
        df_step1.write_csv(DEBUG_DIR / "01_after_phrase_removal.csv")
        logger.info("Saved 01_after_phrase_removal.csv")

        # --- Step 2: Basic Cleaning ---
        logger.info("Running Step 2: Basic Cleaning (lowercase, regex, etc.)")
        df_step2 = df_step1.with_columns(
            pl.col("text")
            .str.replace_all(r"<[^>]+>", " ")
            .str.replace_all(r"c\d+c?", "")
            .str.to_lowercase()
            .str.replace_all(r"[^a-z\s]", "")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("text")
        )
        df_step2.write_csv(DEBUG_DIR / "02_after_basic_cleaning.csv")
        logger.info("Saved 02_after_basic_cleaning.csv")

        # --- Step 3: Stopword Removal ---
        logger.info("Running Step 3: Stopword Removal")
        stopwords_set = DEFAULT_STOPWORDS
        df_step3 = df_step2.with_columns(
            pl.col("text")
            .str.split(" ")
            .list.eval(
                pl.element().filter(
                    ~pl.element().is_in(stopwords_set) & (pl.element() != "")
                )
            )
            .list.join(" ")
            .alias("normalized_text")
        )
        df_step3.write_csv(DEBUG_DIR / "03_after_stopwords.csv")
        logger.info("Saved 03_after_stopwords.csv")
        
        logger.info("--- Final Verification ---")
        filtered = df_step3.filter(pl.col("video_id") == "kOCpYd6-r-A")
        if filtered.height == 0:
            logger.warning("No transcript found for video_id 'kOCpYd6-r-A' after normalization. Skipping final repetition check.")
        else:
            final_text = filtered["normalized_text"][0]
            test_string = "ovide coverage without medical exam"
            if test_string + " " + test_string in final_text:
                logger.error("FAILURE: Repetition still exists in the final output.")
            else:
                logger.info("SUCCESS: Repetition seems to be removed from the final output.")
        logger.info("Debugging script finished. Please check the files in data/debug.")

    except Exception as e:
        logger.error(f"An error occurred during the debug script: {e}", exc_info=True)
