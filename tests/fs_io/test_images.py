import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from src.fs_io.images import write_image, move_image, read_image, delete_images


class TestImagesUtils(unittest.TestCase):

    @patch("src.fs_io.images.logger")
    def test_read_image_not_found(self, mock_logger):
        fake_path = Path("fake/dir/image.png")
        with self.assertRaises(FileNotFoundError):
            read_image(fake_path)
        mock_logger.error.assert_called_once_with(f"Image not found: {fake_path}")

    @patch("src.fs_io.images.cv2.imread")
    @patch("src.fs_io.images.Path.exists", return_value=True)
    @patch("src.fs_io.images.logger")
    def test_read_image_success(self, mock_logger, mock_exists, mock_read_image):
        fake_path = Path("/fake/file.png")
        image_mock = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_read_image.return_value = image_mock

        result = read_image(fake_path)

        mock_read_image.assert_called_once_with(str(fake_path))
        np.testing.assert_array_equal(result, image_mock)


class TestWriteImage(unittest.TestCase):

    @patch("src.fs_io.images.ensure_parent_dir")
    def test_write_image_calls_methods(self, mock_ensure_parent_dir):
        fake_path = Path("/fake/dir/image.png")
        image_bytes = b"fakeimagebytes"
        m = unittest.mock.mock_open()
        with patch("builtins.open", m):
            write_image(image_bytes, fake_path)

        mock_ensure_parent_dir.assert_called_once_with(fake_path)
        m.assert_called_once_with(fake_path, 'wb')
        m().write.assert_called_once_with(image_bytes)

class TestConvertCV2ToPIL(unittest.TestCase):

    @patch("src.fs_io.images.cv2.cvtColor")
    @patch("src.fs_io.images.Image.fromarray")
    def test_cv2_array_to_PIL(self, mock_fromarray, mock_cvtColor):
        from src import cv2_array_to_PIL
        fake_array = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb_array = np.ones((10, 10, 3), dtype=np.uint8)
        mock_cvtColor.return_value = rgb_array
        pil_image_mock = unittest.mock.MagicMock()
        mock_fromarray.return_value = pil_image_mock

        result = cv2_array_to_PIL(fake_array)

        mock_cvtColor.assert_called_once_with(fake_array,  cv2.COLOR_BGR2RGB)
        mock_fromarray.assert_called_once_with(rgb_array)
        self.assertEqual(result, pil_image_mock)

class TestMoveImage(unittest.TestCase):

    @patch("src.fs_io.images.Path.exists", return_value=False)
    @patch("src.fs_io.images.logger")
    def test_move_image_file_not_found(self, mock_logger, mock_exists):
        with self.assertRaises(FileNotFoundError):
            move_image(Path("/fake/image.png"), Path("/target/dir"))
        mock_logger.error.assert_called_once()

    @patch("src.fs_io.images.shutil.move")
    @patch("src.fs_io.images.ensure_dir")
    @patch("src.fs_io.images.Path.exists", return_value=True)
    def test_move_image_success(self, mock_exists, mock_ensure_dir, mock_shutil_move):
        src_path = Path("/fake/image.png")
        target_dir = Path("/target/dir")
        new_path = target_dir / src_path.name

        result = move_image(src_path, target_dir)

        mock_ensure_dir.assert_called_once_with(target_dir)
        mock_shutil_move.assert_called_once_with(src_path, new_path)
        self.assertEqual(result, new_path)


class TestDeleteImages(unittest.TestCase):

    @patch("src.fs_io.images.logger")
    def test_delete_images_force_false(self, mock_logger):
        with self.assertRaises(ValueError):
            delete_images(Path("/fake/dir"), force=False)

    @patch("src.fs_io.images.Path.exists", return_value=False)
    @patch("src.fs_io.images.logger")
    def test_delete_images_nonexistent_dir(self, mock_logger, mock_exists):
        delete_images(Path("/fake/dir"), force=True)
        mock_logger.warning.assert_called_once()

    @patch("src.fs_io.images.Path.exists", return_value=True)
    @patch("src.fs_io.images.Path.is_dir", return_value=True)
    @patch("src.fs_io.images.shutil.rmtree")
    @patch("src.fs_io.images.logger")
    def test_delete_images_success(self, mock_logger, mock_rmtree, mock_is_dir, mock_exists):
        path = Path("/fake/dir")
        delete_images(path, force=True)
        mock_rmtree.assert_called_once_with(path)
        mock_logger.info.assert_called_once_with(f"Deleted all images in directory: {path}")


if __name__ == "__main__":
    unittest.main()