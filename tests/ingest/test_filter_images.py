"""Tests for image filtering heuristics: qrcode, UI detection, and the filtering pipeline that moves/keeps images."""

import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from src.ingest.filter_images import filter_images, is_image_valid, is_ui_element, is_qrcode

TEST_DIR = Path(__file__).parent.parent / "fixtures" / "images"

def read_gray_image(image_path: Path) -> np.ndarray:
    return cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

class TestFilterImages(unittest.TestCase):
    """Verify detection logic for QR codes, UI/gradient images and filtering pipeline behavior."""

    def test_qrcode_detection(self):
        """is_qrcode should detect QR-like patterns in grayscale images."""
        images = {"qrcode.png": True, "real_image.png": False}

        for image, expected in images.items():
            image = read_gray_image(TEST_DIR / image)
            self.assertEqual(is_qrcode(image), expected)

    def test_ui_element_detection(self):
        """is_ui_element should identify UI/gradient images as UI elements (to be filtered)."""
        images = {"ui.png": True, "gradient.png": True, "real_image.png": False}

        for image, expected in images.items():
            image = read_gray_image(TEST_DIR / image)
            self.assertEqual(is_ui_element(image), expected)

    def test_valid_image_detection(self):
        """is_image_valid should mark only true photographic images as valid."""
        images = {
            "clear.png": False, "qrcode.png": False,
            "ui.png": False, "gradient.png": False,
            "real_image.png": True
        }

        for image, expected in images.items():
            image = cv2.imread(str(TEST_DIR / image))
            self.assertEqual(is_image_valid(image).is_valid, expected)

    @patch("src.ingest.filter_images.move_image")
    def test_filter_images_happy_path(self, mock_move_image):
        """filter_images should move invalid images and return only valid images in a dataframe."""
        images = ["clear.png", "qrcode.png", "ui.png", "gradient.png", "real_image.png"]

        count_of_valid_images = 1

        df_images = pd.DataFrame({
            'path': [str(TEST_DIR / image) for image in images],
            'doc_id': [0 for _ in range(len(images))],
            'page': [0 for _ in range(len(images))],
        })

        filtered_images = filter_images(df_images)

        self.assertEqual(mock_move_image.call_count, len(images) - count_of_valid_images)
        self.assertEqual(len(filtered_images), count_of_valid_images)
        self.assertEqual(
            filtered_images.at[0, "path"],
            str(TEST_DIR / "real_image.png")
        )

if __name__ == "__main__":
    unittest.main()