import pandas as pd
import cv2
from pyzbar.pyzbar import decode
import numpy as np
from src import IMAGES_DIR_PATH, REJECTED_IMAGES_DIR_PATH
from src.io.images import move_image
from src.logger import logger

def is_qrcode(gray_image: np.ndarray) -> bool:
    """Check if the grayscale image contains a QR code."""
    detected_objects = decode(gray_image)
    if detected_objects:
        return True

    inverted_image = cv2.bitwise_not(gray_image)
    detected_objects = decode(inverted_image)
    return len(detected_objects) > 0

def is_gradient(gray_image: np.ndarray) -> bool:
    """Check if the grayscale image is mostly gradient/blur (not sharp)."""
    threshold = 100
    h, w = gray_image.shape

    if h < 100 and w < 100:
        return True

    margin_h = int(h * 0.1)
    margin_w = int(w * 0.1)

    if h > 2 * margin_h and w > 2 * margin_w:
        cropped_image = gray_image[margin_h:h - margin_h, margin_w:w - margin_w]
    else:
        return True

    laplacian_var = cv2.Laplacian(cropped_image, cv2.CV_64F).var()
    return laplacian_var < threshold

def filter_images(df_images: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out images that are QR codes or blurry/gradient.
    Moves filtered images to rejected directory.
    Args:
        df_images (pd.DataFrame): DataFrame containing 'path', 'doc_id', 'page', etc.
    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid images.
    """
    filtered = []

    for image_row in df_images.itertuples():
        image_path = IMAGES_DIR_PATH / image_row.path
        image = cv2.imread(image_path)

        if image is None:
            filtered.append(False)
            logger.warning(f"Image could not be read: {image_path}")
        else:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keep_image = not (is_qrcode(gray_image) or is_gradient(gray_image))
            filtered.append(keep_image)

            if not keep_image:
                move_image(image_path, REJECTED_IMAGES_DIR_PATH)

    df_images['filtered'] = filtered
    return df_images[df_images.filtered == True]
