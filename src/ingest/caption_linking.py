import pandas as pd
from math import sqrt

from src import IMAGES_DF_PATH, TEXT_BLOKS_DF_PATH

def find_centroid(bbox: tuple):
    return bbox[2] - bbox[0] / 2, bbox[3] - bbox[1] / 2

def get_distance(first_bbox: tuple, second_bbox: tuple):
    first_centroid = find_centroid(first_bbox)
    second_centroid = find_centroid(second_bbox)

    return sqrt((second_centroid[0] - first_centroid[0]) ** 2
                + (second_centroid[1] - first_centroid[0]) ** 2)

def link_image_to_text():
    images_df = pd.read_pickle(IMAGES_DF_PATH)
    text_blocks_df = pd.read_pickle(TEXT_BLOKS_DF_PATH)

    for index, image_row in images_df.iterrows():

        texts_candidates_df = text_blocks_df[
            (text_blocks_df['doc_id'] == image_row['doc_id']) &
            (text_blocks_df['page'] == image_row['page'])
            ]

        if texts_candidates_df.empty:
            continue

        distance_criteria = lambda second_bbox: get_distance(
            first_bbox=image_row['bbox'],
            second_bbox=second_bbox
        )

        idx = texts_candidates_df['bbox'].apply(distance_criteria).idxmin()
        images_df.loc[index, 'caption'] = texts_candidates_df.loc[idx, 'text']

    images_df.to_pickle(IMAGES_DF_PATH)