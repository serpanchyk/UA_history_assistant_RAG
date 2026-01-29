from enum import Enum
from pathlib import Path

import pandas as pd
import cv2
from pyzbar.pyzbar import decode
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from src import REJECTED_IMAGES_DIR_PATH
from src.fs_io.images import move_image, read_image
from src.logger import logger

class RejectReason(str, Enum):
    READ_FAILED = "read_failed"
    QR_CODE = "qr_code"
    UI_ELEMENT = "ui_element"

THRESHOLD = 90 # Variance of Laplacian threshold for blur detection
H_LIMIT, W_LIMIT = 100, 100 # Minimum height and width to consider for blur detection
CROP_RATIO = 0.1 # Ratio to crop from each side before blur detection

@dataclass(frozen=True)
class ImageValidationResult:
    is_valid: bool
    reasons: list[RejectReason]

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

    h, w = gray_image.shape

    if h < H_LIMIT and w < W_LIMIT:
        return True

    margin_h = int(h * CROP_RATIO)
    margin_w = int(w * CROP_RATIO)

    if h > 2 * margin_h and w > 2 * margin_w:
        cropped_image = gray_image[margin_h:h - margin_h, margin_w:w - margin_w]
    else:
        return True

    laplacian_var = cv2.Laplacian(cropped_image, cv2.CV_64F).var()
    return laplacian_var < THRESHOLD

def is_image_valid(image: np.ndarray) -> ImageValidationResult:
    """Checks whether image is valid: exists, not qrcode and not ui element."""
    reasons = []

    if image is None:
        reasons.append(RejectReason.READ_FAILED)
        return ImageValidationResult(False, reasons)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if is_qrcode(gray_image):
        reasons.append(RejectReason.QR_CODE)
    if is_ui_element(gray_image):
        reasons.append(RejectReason.UI_ELEMENT)

    return ImageValidationResult(len(reasons) == 0, reasons)


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

    for idx in tqdm(
            df_images.index,
            total=len(df_images),
            desc="Filtering images"
    ):
        image_row = df_images.iloc[idx]
        image_path = Path(image_row.path)
        image = read_image(image_path)

        result = is_image_valid(image)

        if result.is_valid:
            keep_indices.append(idx)
        else:
            move_image(image_path, REJECTED_IMAGES_DIR_PATH)

    return df_images.iloc[keep_indices].reset_index(drop=True)
