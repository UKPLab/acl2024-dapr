from __future__ import annotations
from dataclasses import dataclass
import logging
from multiprocessing import Queue
from multiprocessing.context import SpawnProcess
import os
import sys
from typing import Dict, Iterable, List, Optional, Type, Union
from dapr.datasets.dm import Document, Query
from dapr.models.dm import (
    RetrievalLevel,
    RetrievedChunkList,
    RetrievedDocumentList,
    ScoredChunk,
    ScoredDocument,
)
from dapr.models.encoding import ColBERTEncoder, SimilarityFunction, TextEncoder
import multiprocessing as mp
from dapr.models.retrieval.base import (
    ANCEDocumentMethods,
    BaseRetriever,
    DocumentMethod,
    PoolingDocumentMethods,
)
from dapr.utils import NINF, Pooling, Separator, hash_model
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import tqdm


@dataclass
class SearchOutput:
    wid: int  # worker index
    topk_indices: torch.LongTensor  # (nqueries, topk)
    topk_values: torch.Tensor  # (nqueries, topk)
    level: RetrievalLevel

    def update(
        self,
        base_chunk: Optional[int],
        base_doc: Optional[int],
        similarity_matrix: torch.Tensor,
        nqueries: int,
        topk: int,
    ) -> int:
        nentries: int = similarity_matrix.shape[1]  # either #chunks or #documents
        assert similarity_matrix.shape[0] == nqueries
        if self.level is RetrievalLevel.chunk:
            assert base_chunk is not None
            base = base_chunk
        elif self.level is RetrievalLevel.document:
            assert base_doc is not None
            base = base_doc
        else:
            raise NotImplementedError
        self.topk_values, topk_setoffs = torch.cat(
            [self.topk_values, similarity_matrix], dim=-1
        ).topk(k=topk, dim=1, largest=True, sorted=False)
        new_indices = (
            torch.arange(nentries)
            .to(self.topk_indices)
            .unsqueeze(0)
            .expand(nqueries, nentries)
        ) + base
        self.topk_indices = torch.cat([self.topk_indices, new_indices], dim=-1).gather(
            dim=1, index=topk_setoffs
        )
        return nentries

    @classmethod
    def merge(
        cls: Type[SearchOutput], souts: List[SearchOutput], topk: int
    ) -> SearchOutput:
        assert len(souts)
        sout0 = souts[0]
        assert set(map(lambda sout: sout.level, souts)) == {sout0.level}
        topk_values, topk_setoffs = torch.cat(
            [sout.topk_values for sout in souts], dim=-1
        ).topk(k=topk, dim=1, largest=True, sorted=False)
        topk_indices = torch.cat([sout.topk_indices for sout in souts], dim=-1).gather(
            dim=1, index=topk_setoffs
        )
        return SearchOutput(
            wid=-1,
            topk_indices=topk_indices,
            topk_values=topk_values,
            level=sout0.level,
        )

    def to_retrieved_chunk_lists(
        self, queries: List[Query], chunk_ids: List[str], did_per_chunk: List[int]
    ) -> List[RetrievedChunkList]:
        assert len(chunk_ids) == len(did_per_chunk)
        topk_indices = self.topk_indices.cpu().tolist()
        topk_values = self.topk_values.cpu().tolist()
        retrieval_results: List[RetrievedChunkList] = []
        for i, query in enumerate(tqdm.tqdm(queries, desc="Formatting results")):
            scored_chunks: List[ScoredChunk] = []
            cids = map(
                lambda topk_index: chunk_ids[topk_index],
                topk_indices[i],
            )
            dids = map(
                lambda topk_index: did_per_chunk[topk_index],
                topk_indices[i],
            )
            for cid, did, score in zip(cids, dids, topk_values[i]):
                scored_chunks.append(ScoredChunk(chunk_id=cid, doc_id=did, score=score))
            retrieval_results.append(
                RetrievedChunkList(query_id=query.query_id, scored_chunks=scored_chunks)
            )
        return retrieval_results

    def to_retrieved_document_lists(
        self, queries: List[Query], doc_ids: List[int]
    ) -> List[RetrievedDocumentList]:
        topk_indices = self.topk_indices.cpu().tolist()
        topk_values = self.topk_values.cpu().tolist()
        retrieval_results: List[RetrievedDocumentList] = []
        for i, query in enumerate(tqdm.tqdm(queries, desc="Formatting results")):
            scored_documents: List[ScoredDocument] = []
            dids = map(
                lambda topk_index: doc_ids[topk_index],
                topk_indices[i],
            )
            for did, score in zip(dids, topk_values[i]):
                scored_documents.append(ScoredDocument(doc_id=did, score=score))
            retrieval_results.append(
                RetrievedDocumentList(
                    query_id=query.query_id, scored_documents=scored_documents
                )
            )
        return retrieval_results

    def cpu(self) -> SearchOutput:
        return SearchOutput(
            wid=self.wid,
            topk_indices=self.topk_indices.cpu(),
            topk_values=self.topk_values.cpu(),
            level=self.level,
        )


@dataclass
class WorkerInput:
    dbatch: List[Document]
    base_chunk: Optional[int]
    base_doc: Optional[int]


@dataclass
class WorkerOutput:
    search_output: SearchOutput
    nentries: int


class NeuralRetriever(BaseRetriever):
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
        super().__init__(
            name,
            batch_size_query,
            batch_size_chunk,
            bm25_nthreads,
            bm25_weight,
            doc_method,
            query_encoder,
            document_encoder,
        )
        assert query_encoder is not None
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        if document_encoder is None:  # Shared encoder
            self.document_encoder = query_encoder
        device_string = "cuda" if torch.cuda.is_available() else "cpu"
        self.query_encoder.set_device(device_string)
        self.document_encoder.set_device(device_string)
        self.ps: Optional[List[SpawnProcess]] = None

    @property
    def identifier(self) -> str:
        return f"{self.name}_{hash_model(self.query_encoder)}_{hash_model(self.document_encoder)}"

    @property
    def device(self) -> torch.device:
        return self.query_encoder.device

    @property
    def similarity_function(self) -> SimilarityFunction:
        return self.query_encoder.similarity_function

    def watch_alive(self) -> None:
        for p in self.ps:
            if not p.is_alive():
                for p in self.ps:
                    p.terminate()
                self.logger.info("Exit due to error in subprocess")
                sys.exit()

    @staticmethod
    def search_worker(
        wid: int,
        qembs: torch.Tensor,
        qin: Queue,  # <- queue of document batches
        qout: Queue,  # -> queue of IDs and similarity matrix
        encoder: TextEncoder,
        similarity_function: SimilarityFunction,
        batch_size_chunk: int,
        device: torch.device,
        topk: int,
        level: RetrievalLevel,
        doc_pooling: Optional[Pooling] = None,
    ) -> None:
        logger = logging.getLogger(__name__)
        qembs = qembs.to(device)
        encoder.set_device(device)
        search_output = SearchOutput(
            wid=wid,
            topk_indices=torch.zeros(len(qembs), topk, dtype=torch.int64).to(device),
            topk_values=torch.full((len(qembs), topk), NINF).to(device),
            level=level,
        )
        dbatch: List[Document]
        base_chunk: Optional[int]
        base_doc: Optional[int]
        while True:
            try:
                worker_input: WorkerInput = qin.get()
                dbatch, base_chunk, base_doc = (
                    worker_input.dbatch,
                    worker_input.base_chunk,
                    worker_input.base_doc,
                )
                # Similarity matrix:
                encoded = encoder.encode_documents(
                    documents=dbatch, batch_size_chunk=batch_size_chunk
                )
                chunk_embs = encoded.embeddings
                chunk_mask = encoded.mask
                if level is RetrievalLevel.document:
                    assert doc_pooling
                    assert type(encoder) is not ColBERTEncoder
                    beginning_positions = np.cumsum(
                        [0] + [len(doc.candidate_chunk_ids) for doc in dbatch]
                    )
                    spans = zip(beginning_positions, beginning_positions[1:])
                    cembs_padded = pad_sequence(
                        sequences=[chunk_embs[b:e] for b, e in spans], batch_first=True
                    )  # (ndocs, chunk_length, hdim)
                    doc_mask = pad_sequence(
                        sequences=[
                            torch.ones(len(doc.candidate_chunk_ids)) for doc in dbatch
                        ],
                        batch_first=True,
                    ).to(
                        device
                    )  # (ndocs, chunk_length)
                    doc_embeddings = doc_pooling(
                        token_embeddings=cembs_padded, attention_mask=doc_mask
                    )
                    sim_mtrx = similarity_function(
                        query_embeddings=qembs, chunk_embeddings=doc_embeddings
                    )
                    nentires = search_output.update(
                        base_chunk=None,
                        base_doc=base_doc,
                        similarity_matrix=sim_mtrx,
                        nqueries=len(qembs),
                        topk=topk,
                    )
                elif level is RetrievalLevel.chunk:
                    sim_mtrx = similarity_function(
                        query_embeddings=qembs,
                        chunk_embeddings=chunk_embs,
                        chunk_mask=chunk_mask,
                    )
                    nentires = search_output.update(
                        base_chunk=base_chunk,
                        base_doc=None,
                        similarity_matrix=sim_mtrx,
                        nqueries=len(qembs),
                        topk=topk,
                    )
                else:
                    raise NotImplementedError
                qout.put(WorkerOutput(search_output=search_output, nentries=nentires))
            except Exception as e:
                logger.error(str(e))
                raise e

    @torch.no_grad()
    def search(
        self,
        queries: List[Query],
        pool: Iterable[Document],
        ndocs: int,
        nchunks: int,
        topk: int,
        doc_method: Optional[DocumentMethod] = None,
    ) -> Union[List[RetrievedChunkList], List[RetrievedDocumentList]]:
        # Encode queries:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        qembs = self.query_encoder.encode_queries(
            queries=queries,
            batch_size_query=self.batch_size_query,
            show_progress_bar=True,
        )
        doc_pooling: Optional[Pooling] = None
        level: RetrievalLevel = RetrievalLevel.chunk
        total = nchunks
        if doc_method is DocumentMethod.first_p:
            pool = Document.keep_first_chunks_only(
                pool
            )  # TODO: use chunk level also here
            total = ndocs
        elif doc_method in PoolingDocumentMethods:
            doc_pooling = Pooling(self.doc_method.replace("pooling_", ""))
            level = RetrievalLevel.document
            total = ndocs
        dbatches = Document.split_pool(
            pool=pool,
            batch_size_chunk=self.batch_size_chunk,
        )

        # Start workers
        ctx = mp.get_context("spawn")
        ngpus = torch.cuda.device_count()
        # reduce_per = 5000
        reduce_per = 2500
        qin = ctx.Queue(reduce_per)
        qout = ctx.Queue(reduce_per // ngpus)
        self.logger.info(f"Using {ngpus} GPUs")
        self.ps = [
            ctx.Process(
                target=self.search_worker,
                args=(
                    gpu,
                    qembs,
                    qin,
                    qout,
                    self.document_encoder,
                    self.similarity_function,
                    self.batch_size_chunk,
                    torch.device(gpu),
                    topk,
                    level,
                    doc_pooling,
                ),
                daemon=True,
            )
            for gpu in range(ngpus)
        ]
        pbar = tqdm.tqdm(total=total, desc="Searching")
        nput = 0
        for p in tqdm.tqdm(self.ps, desc="Starting processes"):
            p.start()

        # Patch data and reduce results
        wid2sout: Dict[int, SearchOutput] = {}
        chunk_ids: List[str] = []
        did_per_chunk: List[str] = []
        doc_ids: List[str] = []
        try:
            for i, dbatch in enumerate(dbatches):
                qin.put(
                    WorkerInput(
                        dbatch=dbatch, base_chunk=len(chunk_ids), base_doc=len(doc_ids)
                    )
                )
                nput += 1
                for doc in dbatch:
                    doc_ids.append(doc.doc_id)
                    for chunk in doc.chunks:
                        if chunk.chunk_id not in doc.candidate_chunk_ids:
                            continue
                        chunk_ids.append(chunk.chunk_id)
                        did_per_chunk.append(doc.doc_id)

                if (i + 1) % reduce_per == 0 or len(doc_ids) == ndocs:
                    for _ in range(nput):
                        self.watch_alive()
                        worker_output: WorkerOutput = qout.get()
                        wid2sout[
                            worker_output.search_output.wid
                        ] = worker_output.search_output
                        pbar.update(worker_output.nentries)
                    nput = 0  # Remember to empty nput after processing!
        finally:
            for p in self.ps:
                p.kill()

        search_output = SearchOutput.merge(
            list(map(lambda sout: sout.cpu(), wid2sout.values())), topk=topk
        )
        if level is RetrievalLevel.chunk:
            return search_output.to_retrieved_chunk_lists(
                queries=queries, chunk_ids=chunk_ids, did_per_chunk=did_per_chunk
            )
        else:
            return search_output.to_retrieved_document_lists(
                queries=queries, doc_ids=doc_ids
            )

    @torch.no_grad()
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
        return self.search(
            queries=queries,
            pool=pool,
            ndocs=ndocs,
            nchunks=nchunks,
            topk=topk,
            doc_method=None,
        )

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
        if self.doc_method in ANCEDocumentMethods:
            rcls: List[RetrievedChunkList] = self.search(
                queries=queries,
                pool=pool,
                ndocs=ndocs,
                nchunks=nchunks,
                topk=topk,
                doc_method=self.doc_method,
            )
            rdls = [rcl.max_p() for rcl in rcls]
            return rdls
        elif self.doc_method in PoolingDocumentMethods:
            rdls: List[RetrievedDocumentList] = self.search(
                queries=queries,
                pool=pool,
                ndocs=ndocs,
                nchunks=nchunks,
                topk=topk,
                doc_method=self.doc_method,
            )
            return rdls
        else:
            raise NotImplementedError
