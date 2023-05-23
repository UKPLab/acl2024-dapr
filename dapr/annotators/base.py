from abc import ABC, abstractmethod
import os
from typing import Dict, Iterable, TypedDict

from dapr.datasets.base import BaseDataset
from dapr.datasets.dm import Document
from dapr.utils import tqdm_ropen
import ujson


class DocID2DocSummaryJson(TypedDict):
    doc_id: str
    doc_summary: str


class BaseAnnotator(ABC):
    def annotate(self, dataset: BaseDataset, cache_root_dir: str) -> None:
        """Annotate the doc_summary fields inplace."""
        dataset.loaded_data.meta_data["corpus_identifier"] = "/".join(
            [
                dataset.loaded_data.meta_data["corpus_identifier"],
                self.__class__.__name__,
            ]
        )
        cache_fpath = os.path.join(
            cache_root_dir,
            dataset.loaded_data.meta_data["corpus_identifier"],
            "did2dsum.jsonl",
        )
        if os.path.exists(cache_fpath):
            did2dsum: Dict[str, str] = {}
            for line in tqdm_ropen(
                fpath=cache_fpath, desc="Loading document summaries"
            ):
                line_dict: DocID2DocSummaryJson = ujson.loads(line)
                did2dsum[line_dict["doc_id"]] = line_dict["doc_summary"]
        else:
            os.makedirs(os.path.dirname(cache_fpath), exist_ok=True)
            try:
                did2dsum = self._annotate(dataset)
                with open(cache_fpath, "w") as f:
                    for did, dsum in did2dsum.items():
                        line_dict = DocID2DocSummaryJson(doc_id=did, doc_summary=dsum)
                        line = ujson.dumps(line_dict) + "\n"
                        f.write(line)
            except Exception as e:
                if os.path.exists(cache_fpath):
                    os.remove(cache_fpath)
                raise e

        for lqs in [
            dataset.loaded_data.labeled_queries_train,
            dataset.loaded_data.labeled_queries_dev,
            dataset.loaded_data.labeled_queries_test,
        ]:
            if lqs is None:
                continue
            for lq in lqs:
                for jchk in lq.judged_chunks:
                    doc_id = jchk.chunk.belonging_doc.doc_id
                    jchk.chunk.doc_summary = did2dsum[doc_id]

        corpus_iter_fn = dataset.loaded_data.corpus_iter_fn

        def new_corpus_iter_fn() -> Iterable[Document]:
            for doc in corpus_iter_fn():
                for chk in doc.chunks:
                    chk.doc_summary = did2dsum[doc.doc_id]
                yield doc

        dataset.loaded_data.corpus_iter_fn = new_corpus_iter_fn

    @abstractmethod
    def _annotate(self, dataset: BaseDataset) -> Dict[str, str]:
        """Annotate the doc_summary fields. Return a mapping from `doc_id` to `doc_summary`."""
        pass
