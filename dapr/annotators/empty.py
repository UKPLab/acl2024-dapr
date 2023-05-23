from itertools import chain
from typing import Dict
from dapr.annotators.base import BaseAnnotator
from dapr.datasets.base import BaseDataset


class EmptyAnnotator(BaseAnnotator):
    def _annotate(self, dataset: BaseDataset) -> Dict[str, str]:
        data = dataset.loaded_data
        assert data.corpus_iter_fn is not None

        did2dsum: Dict[str, str] = {}
        for doc in data.corpus_iter_fn():
            did2dsum[doc.doc_id] = None

        for lqs in [
            data.labeled_queries_dev,
            data.labeled_queries_test,
            data.labeled_queries_train,
        ]:
            if lqs is None:
                continue
            for lq in lqs:
                for jchk in lq.judged_chunks:
                    for chk in jchk.chunk.belonging_doc.chunks:
                        did2dsum[chk.belonging_doc.doc_id] = None
        return did2dsum
