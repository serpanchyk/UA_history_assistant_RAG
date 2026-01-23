import pandas as pd
from src.utils.spatial_calculations import get_centroid

def bbox_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts text blocks by their spatial coordinates.
    Text blocks are ordered by the vertical coordinate (y) in descending order
    and then by the horizontal coordinate (x) in ascending order, which
    corresponds to a natural reading order on a page.
    Args:
        df (pd.DataFrame): DataFrame containing precomputed x and y centroid
            coordinates.
    Returns:
        pd.DataFrame: Sorted DataFrame.
    """
    return df.sort_values(by=['y', 'x'], ascending=[False, True], ignore_index=True)


def sort_texts(texts_df: pd.DataFrame) -> pd.Series:
    """
    Sorts text blocks in reading order based on bounding box centroids.
    The function computes centroids for each bounding box, adds their x and y
    coordinates to the DataFrame, and then sorts text blocks separately for
    each document and page.
    Args:
        texts_df (pd.DataFrame): DataFrame containing text blocks with a 'bbox'
            column and document identifiers.
    Returns:
        pd.DataFrame: DataFrame with text blocks ordered spatially within
        each document and page.
    """
    centers = texts_df['bbox'].apply(get_centroid)

    texts_df = texts_df.assign(
        x=centers.apply(lambda c: c[0]),
        y=centers.apply(lambda c: c[1]),
    )

    return (
        texts_df
        .groupby(['doc_id', 'page'], group_keys=True)
        .apply(bbox_sort)
        .reset_index(drop=True)
    )