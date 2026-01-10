import pandas as pd
from tqdm import tqdm

from src.logger import logger

def get_centroid(bbox: tuple) -> tuple:
    """
    Computes the centroid of a bounding box.
    Args:
        bbox (tuple): Bounding box coordinates [x0, y0, x1, y1].
    Returns:
        tuple: (x_center, y_center)
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

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

def link_image_to_text(df_images: pd.DataFrame, df_text_blocks: pd.DataFrame) -> pd.DataFrame:
    """
    Links each image to the closest text block on the same page and doc.
    Args:
        df_images (pd.DataFrame): DataFrame containing images with 'bbox', 'page', 'doc_id'.
        df_text_blocks (pd.DataFrame): DataFrame containing text blocks with 'bbox', 'page', 'doc_id'.
    Returns:
        pd.DataFrame: Images DataFrame with 'caption' column added.
    """

    df_images['caption'] = None
    text_groups = df_text_blocks.groupby(['doc_id', 'page'])
    for idx, image_row in enumerate(tqdm(
            df_images.itertuples(),
            total=len(df_images),
            desc="Linking captions to images"
    )):
        key = (image_row.doc_id, image_row.page)

        if key not in text_groups.groups:
            logger.debug(f"No text found for image: {image_row.path}")
            continue

        texts_candidates_df = text_groups.get_group(key)

        distance_criteria = lambda second_bbox: get_distance_squared(
            c1=get_centroid(image_row.bbox),
            c2=get_centroid(second_bbox)
        )

        idx_min = texts_candidates_df['bbox'].apply(distance_criteria).idxmin()
        df_images.iloc[idx, df_images.columns.get_loc('caption')] = texts_candidates_df.loc[idx_min, 'text']

    return df_images
