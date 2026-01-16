def get_centroid(bbox: tuple) -> tuple:
    """
    Computes the centroid of a bounding box.
    Args:
        bbox (tuple): Bounding box coordinates [x0, y0, x1, y1].
    Returns:
        tuple: (x_center, y_center)
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected 4-element bbox, got {len(bbox)}")
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def get_distance_squared(c1: tuple, c2: tuple) -> float:
    """
    Computes squared distance between two points.
    Args:
        c1 (tuple): First point (x, y)
        c2 (tuple): Second point (x, y)
    Returns:
        float: Squared distance
    """
    return (c2[0] - c1[0])**2 + (c2[1] - c1[1])**2

def distance_between_bboxes(bbox1: tuple, bbox2: tuple) -> float:
    """
    Computes the squared distance between the centroids of two bounding boxes.
    Args:
        bbox1 (tuple): Bounding box defined as (x0, y0, x1, y1).
        bbox2 (tuple): Bounding box defined as (x0, y0, x1, y1).
    Returns:
        float: Squared Euclidean distance between the centroids.
    """
    return get_distance_squared(
        c1=get_centroid(bbox1),
        c2=get_centroid(bbox2)
    )
