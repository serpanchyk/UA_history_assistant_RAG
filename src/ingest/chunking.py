import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


class ChunkBuilder:
    """
    Helper class to accumulate text parts into a single chunk.
    Manages the current length and metadata (pages) of the chunk being built.
    """

    def __init__(self, doc_id: int, max_size: int):
        """
        Initializes the ChunkBuilder.
        Args:
            doc_id (int): The identifier of the document being processed.
            max_size (int): The maximum allowed size (in characters) for a chunk.
        """
        self.doc_id = doc_id
        self.max_size = max_size
        self.parts: list[str] = []
        self.current_length = 0
        self.pages: set[int] = set()

    def can_add(self, text_len: int) -> bool:
        """
        Checks if a new text block can fit into the current chunk.
        Args:
            text_len (int): The length of the text block to add.
        Returns:
            bool: True if the text fits within max_size, False otherwise.
        """
        overhead = 1 if self.parts else 0
        return self.current_length + overhead + text_len <= self.max_size

    def add(self, text: str, page: int):
        """
        Adds a text block to the current chunk.
        Args:
            text (str): The text content to add.
            page (int): The page number associated with the text.
        """
        overhead = 1 if self.parts else 0
        self.parts.append(text)
        self.current_length += overhead + len(text)
        self.pages.add(page)

    def flush(self) -> dict | None:
        """
        Finalizes and returns the current chunk, then resets the builder.
        Returns:
            dict | None: A dictionary containing the chunk data if content exists.
        """
        if not self.parts:
            return None

        chunk = {
            "text": " ".join(self.parts),
            "pages": sorted(self.pages),
            "doc_id": self.doc_id,
        }
        self.parts = []
        self.current_length = 0
        self.pages.clear()
        return chunk


def process_document_group(
        doc_id: int,
        group: pd.DataFrame,
        max_chunk_size: int,
        overlap: int
) -> list[dict]:
    """
    Processes all text blocks for a single document.
    Uses LangChain to split large blocks safely with overlap.
    """
    builder = ChunkBuilder(doc_id, max_chunk_size)
    results = []

    # Initialize the splitter with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    for block in group.itertuples(index=False):
        text = block.text
        page = block.page

        if len(text) > max_chunk_size:
            pieces = splitter.split_text(text)
        else:
            pieces = [text]

        for piece in pieces:
            if builder.can_add(len(piece)):
                builder.add(piece, page)
            else:
                if chunk := builder.flush():
                    results.append(chunk)
                builder.add(piece, page)

    if last_chunk := builder.flush():
        results.append(last_chunk)

    return results


def chunking(
        text_blocks_df: pd.DataFrame,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Orchestrates the chunking process for multiple documents.
    """
    result: list[dict] = []

    for doc_id, group in text_blocks_df.groupby("doc_id"):
        doc_chunks = process_document_group(doc_id, group, max_chunk_size, overlap)
        result.extend(doc_chunks)

    return result