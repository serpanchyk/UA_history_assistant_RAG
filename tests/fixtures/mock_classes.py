from typing import NamedTuple

class MockImageRow(NamedTuple):
    path: str
    bbox: list
    doc_id: int
    page: int

class MockPage(NamedTuple):
    number: int