from __future__ import annotations
from enum import Enum


class RetrievalLevel(str, Enum):
    paragraph = "paragraph"
    document = "document"


class ParagraphSeparator(str, Enum):
    blank = "blank"
    newline = "newline"

    @property
    def string(self) -> str:
        return {self.blank: " ", self.newline: "\n"}[self]
