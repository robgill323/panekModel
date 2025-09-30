# src/yt_topics_pro/storage/duck.py
"""
DuckDB connector and convenience functions for querying Parquet data.
"""
import logging
import duckdb

from yt_topics_pro.config import settings

logger = logging.getLogger(__name__)

_con = None


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Initializes and returns a DuckDB connection.
    The connection is cached globally.
    """
    global _con
    if _con is None:
        db_path = settings.storage.duckdb_path
        logger.info(f"Initializing DuckDB connection to: {db_path}")
        _con = duckdb.connect(database=db_path, read_only=False)
    return _con


def register_parquet_views():
    """
    Registers all Parquet files from the configured directory as views in DuckDB.
    """
    con = get_duckdb_connection()
    parquet_path = settings.storage.parquet_dir
    logger.info(f"Registering Parquet files from '{parquet_path}/**/*.parquet' as DuckDB views.")
    
    # This SQL command tells DuckDB to create views for all Parquet files
    # found in the specified directory structure.
    try:
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_transcripts AS SELECT * FROM read_parquet('{parquet_path}/raw/transcripts.parquet');
            CREATE OR REPLACE VIEW raw_metadata AS SELECT * FROM read_parquet('{parquet_path}/raw/metadata.parquet');
            CREATE OR REPLACE VIEW processed_chunks AS SELECT * FROM read_parquet('{parquet_path}/processed/chunks.parquet');
            -- Add other processed tables here as they are created
        """)
        logger.info("Successfully registered Parquet views.")
    except duckdb.Error as e:
        logger.error(f"DuckDB error while registering views: {e}")
        logger.error("This might be because the Parquet files do not exist yet. Run the 'ingest' and 'process' commands first.")


def query_duckdb(query: str, **params) -> duckdb.DuckDBPyRelation:
    """
    Executes a query against the DuckDB database.

    Args:
        query: The SQL query to execute.
        params: Parameters to pass to the query.

    Returns:
        A DuckDB relation object, which can be converted to a DataFrame.
    """
    con = get_duckdb_connection()
    logger.debug(f"Executing DuckDB query: {query}")
    return con.execute(query, params)


if __name__ == "__main__":
    # This file requires Parquet files to be present.
    # See `tables.py` for an example of how to create them.
    print("DuckDB module. Use `get_duckdb_connection` and `register_parquet_views`.")
    
    # Example of how it would be used after running the pipeline:
    # 1. (Run ingest/process to create parquet files)
    # 2. register_parquet_views()
    # 3. result_df = query_duckdb("SELECT * FROM processed_chunks LIMIT 5").pl()
    # 4. print(result_df)
