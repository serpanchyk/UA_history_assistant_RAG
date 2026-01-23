import pandas as pd
from src.utils.spatial_calculations import get_centroid

def sort_texts(texts_df: pd.DataFrame) -> pd.DataFrame:
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

    df = texts_df.copy()

    centers = df['bbox'].apply(get_centroid)
    df[['x', 'y']] = pd.DataFrame(centers.tolist(), index=df.index)

    return (
        df.sort_values(
            by=['doc_id', 'page', 'y', 'x'],
            ascending=[True, True, False, True]
        )
        .reset_index(drop=True)
    )