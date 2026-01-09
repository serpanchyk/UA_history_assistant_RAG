import pandas as pd

from src import IMAGES_DF_PATH, TEXT_BLOKS_DF_PATH

def get_centroid(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def get_distance_squared(c1, c2):
    return (c2[0] - c1[0])**2 + (c2[1] - c1[1])**2

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

        distance_criteria = lambda second_bbox: get_distance_squared(
            c1=get_centroid(image_row['bbox']),
            c2=get_centroid(second_bbox)
        )

        idx = texts_candidates_df['bbox'].apply(distance_criteria).idxmin()
        images_df.loc[index, 'caption'] = texts_candidates_df.loc[idx, 'text']

    images_df.to_pickle(IMAGES_DF_PATH)