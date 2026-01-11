from pathlib import Path

import pandas as pd
import cv2
from pyzbar.pyzbar import decode
import numpy as np
from tqdm import tqdm

from src import REJECTED_IMAGES_DIR_PATH
from src.io.images import move_image, read_image
from src.logger import logger

def is_qrcode(gray_image: np.ndarray) -> bool:
    """Check if the grayscale image contains a QR code."""
    detected_objects = decode(gray_image)
    if detected_objects:
        return True

    inverted_image = cv2.bitwise_not(gray_image)
    detected_objects = decode(inverted_image)
    return len(detected_objects) > 0


def is_ui_element(gray_image: np.ndarray) -> bool:
    """Check if the grayscale image is mostly gradient/blur (not sharp)."""
    if gray_image is None or not isinstance(gray_image, np.ndarray):
        raise ValueError("Input must be a valid numpy ndarray.")

    if len(gray_image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array).")

    threshold = 100
    h_limit, w_limit = 100, 100
    crop_ratio = 0.1

    h, w = gray_image.shape

    if h < h_limit and w < w_limit:
        return True

    margin_h = int(h * crop_ratio)
    margin_w = int(w * crop_ratio)

    if h > 2 * margin_h and w > 2 * margin_w:
        cropped_image = gray_image[margin_h:h - margin_h, margin_w:w - margin_w]
    else:
        return True

    laplacian_var = cv2.Laplacian(cropped_image, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_image_valid(image: np.ndarray, image_path: Path) -> bool:
    """Checks whether image is valid: exists, not qrcode and not ui element."""
    if image is None:
        logger.warning(f"Image could not be read: {image_path}")
        return False

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return not (is_qrcode(gray_image) or is_ui_element(gray_image))


def filter_images(df_images: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out images that are QR codes or blurry/gradient.
    Moves filtered images to rejected directory.
    Args:
        df_images (pd.DataFrame): DataFrame containing 'path', 'doc_id', 'page', etc.
    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid images.
    """
    keep_indices = []

    for idx, image_row in enumerate(tqdm(
            df_images.itertuples(),
            total=len(df_images),
            desc="Filtering images"
    )):
        image_path = Path(image_row.path)
        image = read_image(image_path)

        keep_image = is_image_valid(image, image_path)

        if keep_image:
            keep_indices.append(idx)
        else:
            move_image(image_path, REJECTED_IMAGES_DIR_PATH)


    return df_images.iloc[keep_indices]
