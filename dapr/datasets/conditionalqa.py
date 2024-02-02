from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import re
from typing import List, Optional
from dapr.datasets.dm import (
    Chunk,
    Document,
    LabeledQuery,
    LoadedData,
)
from dapr.utils import set_logger_format
from dapr.datasets.tagged_conditionalqa import TaggedConditionalQA
import tqdm


class ConditionalQA(TaggedConditionalQA):
    """The cleaned version of ConditionalQA where the HTML tags have been removed. Used in the DAPR experiments."""

    HTML_TAG_PATTERN = re.compile("<.*?>")

    def clean_document(self, doc: Document) -> Document:
        cloned_doc = replace(doc)
        chunks = [replace(chk) for chk in cloned_doc.chunks]
        for chk in chunks:
            chk.text = re.sub(self.HTML_TAG_PATTERN, "", chk.text).strip()
        cloned_doc.chunks = chunks
        return cloned_doc

    def clean_labeled_queries(
        self, labeled_queries: List[LabeledQuery]
    ) -> List[LabeledQuery]:
        cleaned_lqs = []
        for lq in tqdm.tqdm(labeled_queries, desc="Cleaning labeled queries"):
            for jchk in lq.judged_chunks:
                jchk.chunk.belonging_doc = self.clean_document(jchk.chunk.belonging_doc)
            cleaned_lqs.append(lq)
        return cleaned_lqs

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        data = super()._load_data(nheldout)
        docs = data.corpus_iter_fn()
        cleaned_docs = [
            self.clean_document(doc) for doc in tqdm.tqdm(docs, desc="Cleaning corpus")
        ]
        return LoadedData(
            corpus_iter_fn=lambda: iter(cleaned_docs),
            labeled_queries_train=self.clean_labeled_queries(
                data.labeled_queries_train
            ),
            labeled_queries_dev=self.clean_labeled_queries(data.labeled_queries_dev),
            labeled_queries_test=self.clean_labeled_queries(data.labeled_queries_test),
        )


if __name__ == "__main__":
    set_logger_format()
    dataset = ConditionalQA(
        resource_path="https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0",
        nheldout=None,
    )
