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
    def test_qrcode_detection(self):
        images = {"qrcode.png": True, "real_image.png": False}

        for image, expected in images.items():
            image = read_gray_image(TEST_DIR / image)
            self.assertEqual(is_qrcode(image), expected)

    def test_ui_element_detection(self):
        images = {"ui.png": True, "gradient.png": True, "real_image.png": False}

        for image, expected in images.items():
            image = read_gray_image(TEST_DIR / image)
            self.assertEqual(is_ui_element(image), expected)

    def test_valid_image_detection(self):
        images = {
            "clear.png": False, "qrcode.png": False,
            "ui.png": False, "gradient.png": False,
            "real_image.png": True
        }

        for image, expected in images.items():
            image = cv2.imread(str(TEST_DIR / image))
            self.assertEqual(is_image_valid(image), expected)

    @patch("src.ingest.filter_images.move_image")
    def test_filter_images_happy_path(self, mock_move_image):
        images = ["clear.png", "qrcode.png", "ui.png", "gradient.png", "real_image.png"]

        index_of_valid_image = 4
        count_of_valid_images = 1

        df_images = pd.DataFrame({
            "path": [str(TEST_DIR / image) for image in images]
        })

        filtered_images = filter_images(df_images)

        self.assertEqual(mock_move_image.call_count, len(images) - count_of_valid_images)
        self.assertEqual(len(filtered_images), count_of_valid_images)
        self.assertEqual(
            filtered_images.at[index_of_valid_image, "path"],
            str(TEST_DIR / "real_image.png")
        )

if __name__ == "__main__":
    unittest.main()