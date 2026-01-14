import pandas as pd

MAX_CHUNK_SIZE = 3800 # calculated size of symbols not to exceed limit of 1024 tokens.

def chunking(
    text_blocks_df: pd.DataFrame,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> list[dict]:

    result: list[dict] = []

    for doc_id, group in text_blocks_df.groupby("doc_id"):
        chunk_parts: list[str] = []
        chunk_len = 0
        pages: set[int] = set()

        for block in group.itertuples(index=False):
            block_text = block.text
            block_len = len(block_text) + 1  # space
            block_page = block.page

            if chunk_len + block_len > max_chunk_size and chunk_parts:
                result.append({
                    "text": " ".join(chunk_parts),
                    "pages": sorted(pages),
                    "doc_id": doc_id,
                })

                chunk_parts = []
                chunk_len = 0
                pages.clear()

            chunk_parts.append(block_text)
            chunk_len += block_len
            pages.add(block_page)

        if chunk_parts:
            result.append({
                "text": " ".join(chunk_parts),
                "pages": sorted(pages),
                "doc_id": doc_id,
            })

    return result