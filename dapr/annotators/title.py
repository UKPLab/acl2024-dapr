from itertools import chain
from typing import Dict
from dapr.annotators.base import BaseAnnotator
from dapr.datasets.base import LoadedData


class TitleAnnotator(BaseAnnotator):
    def _annotate(self, data: LoadedData) -> Dict[str, str]:
        assert data.corpus_iter_fn is not None

        did2dsum: Dict[str, str] = {}
        for doc in data.corpus_iter_fn():
            did2dsum[doc.doc_id] = doc.title

        for lqs in [
            data.labeled_queries_dev,
            data.labeled_queries_test,
            data.labeled_queries_train,
        ]:
            if lqs is None:
                continue
            for lq in lqs:
                for jchk in lq.judged_chunks:
                    doc = jchk.chunk.belonging_doc
                    did2dsum[doc.doc_id] = doc.title
        return did2dsum
