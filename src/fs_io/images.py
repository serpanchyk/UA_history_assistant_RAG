import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.logger import logger
from .filesystem import ensure_dir, ensure_parent_dir


def read_image(path: Path) -> np.ndarray | None:
    """
    Reads an image file and returns its bytes.
    Args:
        path (Path): Path to the image file.
    Returns:
        np.ndarray: The content of the image file.
    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return None

    return cv2.imread(str(path))

def cv2_array_to_PIL(image_array: np.ndarray):
    """
    Converts a CV2 image array (BGR) to a PIL Image (RGB).
    Args:
        image_array (np.ndarray): CV2 image array in BGR format.
    Returns:
        PIL.Image.Image: Converted PIL Image in RGB format.
    """
    # Convert BGR to RGB
    rgb_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)

def write_image(image_bytes: bytes, path: Path) -> None:
    """
    Writes image bytes to the specified path.
    Args:
        image_bytes (bytes): Image content.
        path (Path): Destination path for the image.
    """
    ensure_parent_dir(path)
    with open(path, 'wb') as f:
        f.write(image_bytes)


def move_image(image_path: Path, target_dir: Path) -> Path:
    """
    Moves an image file to a target directory.
    Args:
        image_path (Path): Path to the image to move.
        target_dir (Path): Directory where the image will be moved.
    Returns:
        Path: New path of the moved image.
    Raises:
        FileNotFoundError: If the source image does not exist.
    """
    if not image_path.exists():
        logger.error(f"Image not found for moving: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")

    ensure_dir(target_dir)
    new_path = target_dir / image_path.name
    shutil.move(image_path, new_path)

    return new_path


def delete_images(images_path: Path, force: bool = False) -> None:
    """
    Deletes all images in the specified directory.
    Args:
        images_path (Path): Path to the directory containing images.
    Notes:
        If the path does not exist or is not a directory, the function does nothing.
    """
    if not force:
        raise ValueError("Must pass force=True to delete images")

    if not images_path.exists():
        logger.warning(f"Directory to delete does not exist: {images_path}")
        return

    if not images_path.is_dir():
        logger.warning(f"Path to delete is not a directory: {images_path}")
        return

    shutil.rmtree(images_path)
    logger.info(f"Deleted all images in directory: {images_path}")