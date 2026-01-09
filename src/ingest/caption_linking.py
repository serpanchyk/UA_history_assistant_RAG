import pandas as pd


def get_centroid(bbox: tuple) -> tuple:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def get_distance_squared(c1: tuple, c2: tuple) -> float:
    return (c2[0] - c1[0])**2 + (c2[1] - c1[1])**2

def link_image_to_text(df_images: pd.DataFrame, df_text_blocks: pd.DataFrame) -> pd.DataFrame:

    for image_row in df_images.itertuples():

        texts_candidates_df = df_text_blocks[
            (df_text_blocks['doc_id'] == image_row.doc_id) &
            (df_text_blocks['page'] == image_row.page)
            ]

        if texts_candidates_df.empty:
            continue

        distance_criteria = lambda second_bbox: get_distance_squared(
            c1=get_centroid(image_row.bbox),
            c2=get_centroid(second_bbox)
        )

        idx = texts_candidates_df['bbox'].apply(distance_criteria).idxmin()
        df_images.loc[image_row.path, 'caption'] = texts_candidates_df.loc[idx, 'text']

    return df_images