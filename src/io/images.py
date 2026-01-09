import shutil
from pathlib import Path

from .filesystem import ensure_parent_dir

def read_image(path: Path) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path.read_bytes()

def write_image(image_bytes: bytes, path: Path) -> None:
    ensure_parent_dir(path)

    with open(path, 'wb') as f:
        f.write(image_bytes)

def move_image(image_path: Path, target_dir: Path) -> Path:

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ensure_parent_dir(target_dir)

    new_path = target_dir / image_path.name

    shutil.move(image_path, new_path)

    return new_path

def delete_images(images_path: Path) -> None:
    if not images_path.exists():
        return

    if not images_path.is_dir():
        return


    shutil.rmtree(images_path)
