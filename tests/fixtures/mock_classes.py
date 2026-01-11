from typing import NamedTuple

class MocImageRow(NamedTuple):
    path: str
    bbox: list
    doc_id: int
    page: int