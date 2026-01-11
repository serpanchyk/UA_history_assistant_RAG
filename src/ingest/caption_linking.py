import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from tqdm import tqdm

from src.logger import logger
from src.utils.distance import distance_between_bboxes

def get_texts_for_image(image_row: tuple, text_groups: DataFrameGroupBy) -> pd.DataFrame | None:
    """
    Retrieves all text blocks that belong to the same document and page
    as the given image.
    Args:
        image_row (tuple): A row from the images DataFrame containing
            at least 'doc_id', 'page', and 'path'.
        text_groups (DataFrameGroupBy): Text blocks grouped by ('doc_id', 'page').
    Returns:
        pd.DataFrame | None: DataFrame of text blocks for the image's page,
        or None if no text blocks are found.
    """

    key = (image_row.doc_id, image_row.page)

    if key not in text_groups.groups:
        logger.debug(f"No text found for image: {image_row.path}")
        return None

    return text_groups.get_group(key)


def find_closest_text(image_bbox: tuple, texts_df: pd.DataFrame) -> str | None:
    """
    Finds the text block whose bounding box is closest to the given image.
    Distance is computed between bounding box centroids.
    Args:
        image_bbox (tuple): Bounding box of the image (x0, y0, x1, y1).
        texts_df (pd.DataFrame): DataFrame containing text blocks with
            'bbox' and 'text' columns.
    Returns:
        str | None: Text of the closest text block, or None if no text is available.
    """
    if texts_df.empty:
        return None

    distances = texts_df['bbox'].apply(
        lambda bbox: distance_between_bboxes(image_bbox, bbox)
    )

    idxmin = distances.idxmin()

    return texts_df.loc[idxmin, 'text']

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
    for image_row in tqdm(
            df_images.itertuples(),
            total=len(df_images),
            desc="Linking captions to images"
    ):
        texts_df = get_texts_for_image(image_row, text_groups)
        if texts_df is None:
            continue

        caption = find_closest_text(tuple(image_row.bbox), texts_df)
        df_images.at[image_row.Index, 'caption'] = caption

    return df_images