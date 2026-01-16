"""Tests for geometric helpers used for centroid and distance computations between bboxes/points."""

import unittest

from src.utils.spatial_calculations import get_centroid, get_distance_squared, distance_between_bboxes


class TestGetCentroid(unittest.TestCase):
    """Verify centroid calculation for bounding boxes, including negative coordinates."""

    def test_basic_bbox(self):
        """get_centroid should compute expected center for a basic bbox."""
        bbox = (0, 0, 10, 20)
        self.assertEqual(get_centroid(bbox), (5, 10))

    def test_negative_coords(self):
        """Centroid logic should correctly handle negative coordinates and return middle point."""
        bbox = (-10, -10, 10, 10)
        self.assertEqual(get_centroid(bbox), (0, 0))


class TestGetDistanceSquared(unittest.TestCase):
    """Unit tests for squared distance between 2D points (used for nearest-text heuristics)."""

    def test_pythagoras(self):
        """get_distance_squared should return squared distance following Pythagoras (3,4 -> 25)."""
        self.assertEqual(
            get_distance_squared((0, 0), (3, 4)),
            25
        )

    def test_same_point(self):
        """Distance squared between identical points should be zero."""
        self.assertEqual(
            get_distance_squared((5, 5), (5, 5)),
            0
        )

class TestDistanceBetweenBboxes(unittest.TestCase):
    """Ensure centroid-based distance between bboxes is computed as expected."""

    def test_centroid_distance(self):
        """distance_between_bboxes returns squared distance between centroids of two boxes."""
        bbox1 = (0, 0, 10, 10)     # (5,5)
        bbox2 = (10, 10, 20, 20)   # (15,15)
        self.assertEqual(
            distance_between_bboxes(bbox1, bbox2),
            200
        )

if __name__ == '__main__':
    unittest.main()