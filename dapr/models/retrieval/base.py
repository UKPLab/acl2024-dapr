from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
import logging
import os
from typing import Dict, Iterable, List, Optional

from dapr.datasets.dm import Chunk, Document, Query
from dapr.models.dm import (
    RetrievalLevel,
    RetrievedChunkList,
    RetrievedChunkListJson,
    RetrievedDocumentList,
    RetrievedDocumentListJson,
)
from dapr.utils import Separator, md5
from dapr.models.encoding import TextEncoder
import ujson


class DocumentMethod(str, Enum):
    max_p = "max_p"
    first_p = "first_p"
    pooling_max = "pooling_max"  # Max-pooling over chunk embeddings
    pooling_mean = "pooling_mean"
    pooling_sum = "pooling_sum"


ANCEDocumentMethods = [DocumentMethod.max_p, DocumentMethod.first_p]

PoolingDocumentMethods = [
    DocumentMethod.pooling_max,
    DocumentMethod.pooling_mean,
    DocumentMethod.pooling_sum,
]


class BaseRetriever(ABC):
    def __init__(
        self,
        name: str,
        batch_size_query: int,
        batch_size_chunk: int,
        bm25_nthreads: Optional[int] = 12,
        bm25_weight: Optional[float] = None,
        doc_method: Optional[DocumentMethod] = None,
        query_encoder: Optional[TextEncoder] = None,
        document_encoder: Optional[TextEncoder] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.batch_size_query = batch_size_query
        self.batch_size_chunk = batch_size_chunk
        self.doc_method = doc_method

    @abstractproperty
    def identifier(self) -> str:
        pass

    def build_retrieval_results_identifier(
        self,
        queries: List[Query],
        pool_identifier,
        level: RetrievalLevel,
        topk: int,
        doc_method: Optional[DocumentMethod] = None,
    ) -> str:
        query_texts = map(lambda query: query.text, queries)
        queries_identifier = f"{md5(query_texts)}_{len(queries)}"
        return "-".join(
            [
                pool_identifier,
                self.identifier,
                f"{level}_{doc_method}" if doc_method else level,
                queries_identifier,
                f"topk_{topk}",
            ]
        )

    def retrieve_chunk_lists(
        self,
        queries: List[Query],
        pool: Iterable[Document],
        ndocs: int,
        nchunks: int,
        pool_identifier: str,
        chunk_separator: Separator,
        topk: int,
        results_dir: str,
        sort_pool: bool = True,
    ) -> List[RetrievedChunkList]:
        rrs_identifier = self.build_retrieval_results_identifier(
            queries=queries,
            pool_identifier=pool_identifier,
            level=RetrievalLevel.chunk,
            topk=topk,
        )
        fpath_rrs = os.path.join(results_dir, f"{rrs_identifier}.jsonl")
        if os.path.exists(fpath_rrs):
            self.logger.info(f"Loading RCLs from {fpath_rrs}")
            rcls = []
            with open(fpath_rrs) as f:
                for line in f:
                    rcl_json: RetrievedChunkListJson = ujson.loads(line)
                    rcls.append(RetrievedChunkList.from_json(rcl_json))
        else:
            rcls = self._retrieve_chunk_lists(
                queries=queries,
                pool=pool,
                ndocs=ndocs,
                nchunks=nchunks,
                pool_identifier=pool_identifier,
                chunk_separator=chunk_separator,
                topk=topk,
                sort_pool=sort_pool,
            )
            os.makedirs(os.path.dirname(fpath_rrs), exist_ok=True)
            try:
                with open(fpath_rrs, "w") as f:
                    for rcl in rcls:
                        line = ujson.dumps(rcl.to_json()) + "\n"
                        f.write(line)
            except Exception as e:
                if os.path.exists(fpath_rrs):
                    os.remove(fpath_rrs)
                raise e
            self.logger.info(f"Saved RCLs to {fpath_rrs}")
        return rcls

    @abstractmethod
    def _retrieve_chunk_lists(
        self,
        queries: List[Query],
        pool: Iterable[Document],
        ndocs: int,
        nchunks: int,
        pool_identifier: str,
        chunk_separator: Separator,
        topk: int,
        sort_pool: bool = True,
    ) -> List[RetrievedChunkList]:
        pass

    def retrieve_document_lists(
        self,
        queries: List[Query],
        pool: Iterable[Document],
        ndocs: int,
        nchunks: int,
        pool_identifier: str,
        chunk_separator: Separator,
        topk: int,
        results_dir: str,
        sort_pool: bool = True,
    ) -> List[RetrievedDocumentList]:
        rrs_identifier = self.build_retrieval_results_identifier(
            queries=queries,
            pool_identifier=pool_identifier,
            level=RetrievalLevel.document,
            topk=topk,
            doc_method=self.doc_method,
        )
        fpath_rrs = os.path.join(results_dir, f"{rrs_identifier}.jsonl")
        if os.path.exists(fpath_rrs):
            self.logger.info(f"Loading RDLs from {fpath_rrs}")
            rdls = []
            with open(fpath_rrs) as f:
                for line in f:
                    rdl_json: RetrievedDocumentListJson = ujson.loads(line)
                    rdls.append(RetrievedDocumentList.from_json(rdl_json))
        else:
            rdls = self._retrieve_document_lists(
                queries=queries,
                pool=pool,
                ndocs=ndocs,
                nchunks=nchunks,
                pool_identifier=pool_identifier,
                chunk_separator=chunk_separator,
                topk=topk,
                sort_pool=sort_pool,
            )
            os.makedirs(os.path.dirname(fpath_rrs), exist_ok=True)
            try:
                with open(fpath_rrs, "w") as f:
                    for rdl in rdls:
                        line = ujson.dumps(rdl.to_json()) + "\n"
                        f.write(line)
            except Exception as e:
                if os.path.exists(fpath_rrs):
                    os.remove(fpath_rrs)
                raise e
            self.logger.info(f"Saved RDLs to {fpath_rrs}")
        return rdls

    @abstractmethod
    def _retrieve_document_lists(
        self,
        queries: List[Query],
        pool: Iterable[Document],
        ndocs: int,
        nchunks: int,
        pool_identifier: str,
        chunk_separator: Separator,
        topk: int,
        sort_pool: bool = True,
    ) -> List[RetrievedDocumentList]:
        pass
