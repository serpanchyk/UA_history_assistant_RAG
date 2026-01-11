import unittest

from src.utils.distance import get_centroid, get_distance_squared, distance_between_bboxes


class TestGetCentroid(unittest.TestCase):

    def test_basic_bbox(self):
        bbox = (0, 0, 10, 20)
        self.assertEqual(get_centroid(bbox), (5, 10))

    def test_negative_coords(self):
        bbox = (-10, -10, 10, 10)
        self.assertEqual(get_centroid(bbox), (0, 0))


class TestGetDistanceSquared(unittest.TestCase):

    def test_pythagoras(self):
        self.assertEqual(
            get_distance_squared((0, 0), (3, 4)),
            25
        )

    def test_same_point(self):
        self.assertEqual(
            get_distance_squared((5, 5), (5, 5)),
            0
        )

class TestDistanceBetweenBboxes(unittest.TestCase):

    def test_centroid_distance(self):
        bbox1 = (0, 0, 10, 10)     # (5,5)
        bbox2 = (10, 10, 20, 20)   # (15,15)
        self.assertEqual(
            distance_between_bboxes(bbox1, bbox2),
            200
        )

if __name__ == '__main__':
    unittest.main()