from abc import ABC, abstractmethod
import inspect
import json
import os
from random import Random
import shutil
from typing import Any, Dict, Iterable, List, Optional
from dapr.utils import Separator, md5
from dapr.datasets.dm import Document, LabeledQuery, LoadedData
import numpy as np
from transformers import AutoTokenizer
import logging


class BaseDataset(ABC):
    def __init__(
        self,
        resource_path: str,
        nheldout: Optional[int],
        cache_root_dir: str = "data",
        chunk_separator: Separator = Separator.empty,
        tokenizer: str = "roberta-base",
        nprocs: int = 10,
    ) -> None:
        self.kwargs = dict(inspect.getargvalues(inspect.currentframe()).locals)
        self.kwargs.pop("self")
        self.kwargs.pop("nprocs")
        self.logger = logging.getLogger(__name__)
        self.resource_path = resource_path
        self.chunk_separator = chunk_separator
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.nprocs = nprocs
        cache_dir = os.path.join(cache_root_dir, self.name)
        if os.path.exists(cache_dir) and os.listdir(cache_dir):
            loaded_data = LoadedData.from_dump(cache_dir)
        else:
            self._download(resource_path)
            loaded_data = self.load_data(nheldout=nheldout, cache_dir=cache_dir)
        self.loaded_data = loaded_data

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def estimate_chunk_length(
        self, corpus: Iterable[Document], n: int = 1000, candidates_only: bool = False
    ) -> float:
        if candidates_only:
            chunks = [
                chk
                for doc in corpus
                for chk in doc.chunks
                if chk.chunk_id in doc.candidate_chunk_ids
            ]
        else:
            chunks = [chk for doc in corpus for chk in doc.chunks]
        random_state = Random(42)
        sampled = random_state.sample(chunks, k=min(n, len(chunks)))
        lengths = [len(self.tokenizer.tokenize(chunk.text)) for chunk in sampled]
        return float(np.mean(lengths))

    def stats(self, loaded_data: LoadedData) -> Dict[str, Any]:
        assert loaded_data.corpus_iter_fn is not None
        corpus_size = sum(1 for _ in loaded_data.corpus_iter_fn())
        assert loaded_data.labeled_queries_test is not None
        stats = {
            "name": self.name,
            "ndocs": corpus_size,
            "nchunks": Document.nchunks_in_corpus(loaded_data.corpus_iter_fn()),
            "nchunks_candidates": Document.nchunks_in_corpus(
                loaded_data.corpus_iter_fn(), candidates_only=True
            ),
            "nchunks_percentiles": Document.nchunks_percentiles(
                corpus=loaded_data.corpus_iter_fn()
            ),
            "nchunks_candidates_percentiles": Document.nchunks_percentiles(
                corpus=loaded_data.corpus_iter_fn(), candidates_only=True
            ),
            "avg_chunk_length": self.estimate_chunk_length(
                loaded_data.corpus_iter_fn()
            ),
            "avg_candidate_chunk_length": self.estimate_chunk_length(
                loaded_data.corpus_iter_fn(), candidates_only=True
            ),
        }
        for split, lqs in [
            ("train", loaded_data.labeled_queries_train),
            ("dev", loaded_data.labeled_queries_dev),
            ("test", loaded_data.labeled_queries_test),
        ]:
            stats[f"nqueries_{split}"] = LabeledQuery.nqueries(lqs) if lqs else None
            stats[f"jpq_{split}"] = LabeledQuery.njudged_per_query(lqs) if lqs else None
        stats = {
            k: round(v, 2) if type(v) in [int, float] else v for k, v in stats.items()
        }
        self.logger.info("\n" + json.dumps(stats, indent=4))
        return stats

    def check_heldout_in_corpus(self, loaded_data: LoadedData) -> None:
        """Assert that the chunk ids in heldout data are all included in the corpus."""
        cids_coprus = set(
            [chk.chunk_id for doc in loaded_data.corpus_iter_fn() for chk in doc.chunks]
        )
        cids_heldout = set()
        assert loaded_data.labeled_queries_test is not None
        lqs: List[LabeledQuery]
        for lqs in [
            loaded_data.labeled_queries_dev,
            loaded_data.labeled_queries_test,
        ]:
            if lqs is None:
                continue
            for lq in lqs:
                for jc in lq.judged_chunks:
                    cids_heldout.add(jc.chunk.chunk_id)

        cids_left = cids_heldout - cids_coprus
        assert len(cids_heldout - cids_coprus) == 0, f"Left: {cids_left}"

    @abstractmethod
    def _download(self, resource_path: str) -> None:
        pass

    def load_data(self, nheldout: Optional[int], cache_dir: str) -> LoadedData:
        loaded_data = self._load_data(nheldout)
        loaded_data.meta_data = {}
        loaded_data.meta_data["chunk_separator"] = self.chunk_separator
        loaded_data.meta_data["corpus_identifier"] = (
            f"{self.name}_{md5(map(lambda doc: str(doc.to_json()), loaded_data.corpus_iter_fn()))}"
        )
        stats = self.stats(loaded_data)
        loaded_data.meta_data.update(stats)
        self.check_heldout_in_corpus(loaded_data)
        try:
            loaded_data.dump(cache_dir)
        except Exception as e:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            raise e
        del loaded_data
        return LoadedData.from_dump(cache_dir)

    @abstractmethod
    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        pass


# Stats:
# {
#     "name": "Genomics",
#     "#docs": 162259,
#     "#chks": 4035411,
#     "#chks percentiles": {
#         "5": 5.0,
#         "25": 19.0,
#         "50": 25.0,
#         "75": 32.0,
#         "95": 41.0
#     },
#     "#train queries": null,
#     "#Judged per query (train)": null,
#     "#dev queries": null,
#     "#Judged per query (dev)": null,
#     "#test queries": 62,
#     "#Judged per query (test)": 225.52
# }
# {
#     "name": "MSMARCO",
#     "#docs": 3201821,
#     "#chks": 11799171,
#     "percentiles": {
#         "5": 1.0,
#         "25": 1.0,
#         "50": 2.0,
#         "75": 4.0,
#         "95": 11.0
#     },
#     "#train queries": 170025,
#     "#Judged per query (train)": 1.09,
#     "#dev queries": 25908,
#     "#Judged per query (dev)": 1.09,
#     "#test queries": 25908,
#     "#Judged per query (test)": 1.1
# }
# {
#     "name": "NaturalQuestions",
#     "#docs": 108626,
#     "#chks": 593737,
#     "#chks percentiles": {
#         "5": 1.0,
#         "25": 2.0,
#         "50": 3.0,
#         "75": 7.0,
#         "95": 18.0
#     },
#     "#train queries": 93275,
#     "#Judged per query (train)": 1.11,
#     "#dev queries": 3610,
#     "#Judged per query (dev)": 1.09,
#     "#test queries": 3610,
#     "#Judged per query (test)": 1.27
# }
