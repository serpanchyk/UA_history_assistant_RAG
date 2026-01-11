from pathlib import Path
from src.logger import logger


def ensure_dir(path: Path) -> None:
    """
    Ensures that a directory exists. Creates it if it does not exist.
    Args:
        path (Path): Directory path to ensure.
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


def ensure_parent_dir(file_path: Path) -> None:
    """
    Ensures that the parent directory of a file exists.
    Args:
        file_path (Path): Path to the file whose parent directory should exist.
    """
    ensure_dir(file_path.parent)
