import pandas as pd

from pathlib import Path

from src.io.filesystem import ensure_parent_dir

def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_parquet(path)