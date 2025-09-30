import polars as pl
from pathlib import Path

# Assuming the script is run from the project root
processed_file = Path("data/parquet/processed/chunks.parquet")

if processed_file.exists():
    try:
        df = pl.read_parquet(processed_file)
        print("File found. Columns are:")
        print(df.columns)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
else:
    print(f"File not found at: {processed_file}")
