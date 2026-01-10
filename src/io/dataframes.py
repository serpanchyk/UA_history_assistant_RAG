import pandas as pd
from pathlib import Path
from src.io.filesystem import ensure_parent_dir
from src.logger import logger


def read_parquet(path: Path) -> pd.DataFrame:
    """
    Reads a Parquet file into a DataFrame.
    Args:
        path (Path): Path to the Parquet file.
    Returns:
        pd.DataFrame: DataFrame with the file contents.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        logger.error(f"Parquet file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Reading Parquet file: {path}")
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Writes a DataFrame to a Parquet file.
    Args:
        df (pd.DataFrame): DataFrame to write.
        path (Path): Destination path for the Parquet file.
    """
    ensure_parent_dir(path)
    df.to_parquet(path)
    logger.info(f"Wrote DataFrame to Parquet file: {path}")
