from pathlib import Path

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def ensure_parent_dir(file_path: Path) -> None:
    ensure_dir(file_path.parent)