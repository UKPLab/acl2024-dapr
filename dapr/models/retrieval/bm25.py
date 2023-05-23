from __future__ import annotations
from dataclasses import dataclass
import json
from multiprocessing.pool import ThreadPool
import os
import shutil
from typing import Callable, Dict, Iterable, List, Optional, Type
from dapr.models.encoding import TextEncoder

import tqdm

from dapr.datasets.dm import Chunk, Document, Query
from dapr.models.retrieval.base import BaseRetriever, DocumentMethod
from dapr.models.dm import (
    RetrievedChunkList,
    RetrievedDocumentList,
    ScoredChunk,
    ScoredDocument,
)
from dapr.utils import Separator, sha256
import tempfile


@dataclass
class PyseriniHit:
    docid: str
    score: float

    @classmethod
    def from_pyserini(cls: Type[PyseriniHit], hit: PyseriniHit) -> PyseriniHit:
        """This conversion step makes the memory releasable (otherwise leaked by Java)."""
        return PyseriniHit(docid=hit.docid, score=hit.score)


@dataclass
class PyseriniCollectionRow:
    id: str
    title: str
    contents: str

    def to_json(self) -> Dict[str, str]:
        return {"id": self.id, "title": self.title, "contents": self.contents}


class BM25Retriever(BaseRetriever):
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
        os.environ[
            "_JAVA_OPTIONS"
        ] = "-Xmx5g"  # Otherwise it would cause to huge memory leak!
        self.bm25_weight = bm25_weight
        self.bm25_nthreads = bm25_nthreads
        self.thread_pool = ThreadPool(processes=bm25_nthreads)
        self.fields = {
            "contents": 1.0,
            "title": 1.0,
        }  # chunk (cannot use other names for this)
        self.cache_path: Optional[str] = None
        self.index_path_chunks: Optional[str] = None
        self.index_path_documents: Optional[str] = None
        self.cid2did: Optional[Dict[str, str]] = None

    @property
    def identifier(self) -> str:
        return self.name  # bm25

    def clear_cache(self) -> None:
        if self.cache_path is not None:
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)
                self.logger.info(f"Removed BM25 cache path {self.cache_path}")

    def index(
        self,
        rows: Iterable[PyseriniCollectionRow],
        identifier: str,
        total: Optional[int] = None,
    ) -> str:
        cache_path = os.path.join(tempfile.gettempdir(), "pyserini", sha256(identifier))
        self.logger.info(f"BM25 cache path: {cache_path}")
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
        index_path = os.path.join(cache_path, "index")
        if os.path.exists(index_path):
            if os.listdir(index_path):
                self.logger.info(f"Found existing index {index_path}")
                list(rows)  # To get cid2did
                return index_path

        import pyserini.index.lucene
        from jnius import autoclass

        coprus_path = os.path.join(cache_path, "corpus")
        os.makedirs(coprus_path, exist_ok=True)
        with open(os.path.join(coprus_path, "texts.jsonl"), "w") as f:
            for row in tqdm.tqdm(
                rows,
                total=total,
                desc="Converting to pyserini format",
            ):
                line = row.to_json()
                f.write(json.dumps(line) + "\n")

        args = [
            "-collection",
            "JsonCollection",
            "-generator",
            "DefaultLuceneDocumentGenerator",
            "-threads",
            str(self.bm25_nthreads),
            "-input",
            coprus_path,
            "-index",
            index_path,
            # "-storeRaw",
            "-storePositions",
            "-storeDocvectors",
            "-fields",
            "title",
        ]
        JIndexCollection = autoclass("io.anserini.index.IndexCollection")
        index_fn: Callable[
            [
                List[str],
            ],
            None,
        ] = getattr(JIndexCollection, "main")
        try:
            index_fn(args)
        except Exception as e:
            shutil.rmtree(index_path)
            raise e

        return index_path

    def index_chunks(
        self, pool: Iterable[Document], nchunks: int, pool_identifier: str
    ) -> str:
        self.logger.info("Indexing chunks")
        self.cid2did = {}

        def rows() -> Iterable[PyseriniCollectionRow]:
            for doc in pool:
                for chk in doc.chunks:
                    title = chk.doc_summary if chk.doc_summary else ""
                    self.cid2did[chk.chunk_id] = doc.doc_id
                    yield PyseriniCollectionRow(
                        id=chk.chunk_id, title=title, contents=chk.text
                    )

        pool_identifier = "/".join([pool_identifier, "chunks"])
        index_path = self.index(rows=rows(), identifier=pool_identifier, total=nchunks)
        self.index_path_chunks = index_path
        from pyserini.index.lucene import IndexReader

        self.index_reader_chunks = IndexReader(index_path)
        return index_path

    def index_documents(
        self,
        pool: List[Document],
        ndocs: int,
        pool_identifier: str,
        chunk_separator: Separator,
    ) -> str:
        self.logger.info("Indexing documents")

        def rows() -> Iterable[PyseriniCollectionRow]:
            for doc in pool:
                text = chunk_separator.concat(map(lambda chk: chk.text, doc.chunks))
                title = doc.chunks[0].doc_summary
                if title is None:
                    title = ""
                yield PyseriniCollectionRow(id=doc.doc_id, title=title, contents=text)

        pool_identifier = "/".join([pool_identifier, "docs"])
        index_path = self.index(
            rows=rows(),
            identifier=pool_identifier,
            total=ndocs,
        )
        self.index_path_documents = index_path
        from pyserini.index.lucene import IndexReader

        self.index_reader_documents = IndexReader(index_path)
        return index_path

    def search(
        self, queries: List[Query], index_path: str, topk: int
    ) -> Dict[str, List[PyseriniHit]]:
        from pyserini.search import SimpleSearcher

        searcher = SimpleSearcher(index_path)
        searcher.set_bm25()
        qid2hits_all = {}
        for b in tqdm.trange(
            0, len(queries), self.batch_size_query, desc="Query batch"
        ):
            e = b + self.batch_size_query
            qid2hits = searcher.batch_search(
                queries=list(map(lambda query: query.text, queries[b:e])),
                qids=list(map(lambda query: query.query_id, queries[b:e])),
                k=topk,
                threads=self.bm25_nthreads,
                fields=self.fields,
            )
            qid2hits_converted = {
                qid: list(map(PyseriniHit.from_pyserini, hits))
                for qid, hits in qid2hits.items()
            }
            qid2hits_all.update(qid2hits_converted)
        searcher.close()
        return qid2hits_all  # Note this might lose the query order!

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
        # Chunk index and search:
        self.index_chunks(pool=pool, nchunks=nchunks, pool_identifier=pool_identifier)
        assert self.cid2did
        qid2hits = self.search(
            queries=queries, index_path=self.index_path_chunks, topk=topk
        )

        # Build retrieval results:
        retrieval_results = []
        for query in queries:
            hits = qid2hits[query.query_id]
            scored_chunks = []
            for hit in hits:
                scored_chunks.append(
                    ScoredChunk(
                        chunk_id=hit.docid,
                        doc_id=self.cid2did[hit.docid],
                        score=hit.score,
                    )
                )
            retrieval_results.append(
                RetrievedChunkList(query_id=query.query_id, scored_chunks=scored_chunks)
            )
        return retrieval_results

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
        # Chunk index and search:
        self.index_documents(
            pool=pool,
            ndocs=ndocs,
            pool_identifier=pool_identifier,
            chunk_separator=chunk_separator,
        )
        qid2hits = self.search(
            queries=queries, index_path=self.index_path_documents, topk=topk
        )

        # Build retrieval results:
        retrieval_results = []
        for query in queries:
            hits = qid2hits[query.query_id]
            scored_documents = []
            for hit in hits:
                scored_documents.append(
                    ScoredDocument(doc_id=hit.docid, score=hit.score)
                )
            retrieval_results.append(
                RetrievedDocumentList(
                    query_id=query.query_id, scored_documents=scored_documents
                )
            )
        return retrieval_results
