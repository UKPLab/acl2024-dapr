"""Data models."""
from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from itertools import chain
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    Union,
)
from enum import Enum
from dapr.utils import Separator
import ujson
import tqdm
import numpy as np


class Split(str, Enum):
    """Dataset split."""

    train = "train"
    dev = "dev"
    test = "test"


class ChunkJson(TypedDict):
    id: str
    text: str


@dataclass
class Chunk:
    """Smallest unit in a corpus."""

    chunk_id: str
    text: str
    doc_summary: Optional[Union[str, np.ndarray]]
    belonging_doc: Document

    @staticmethod
    def doc_summary_str(doc_summary: Optional[Union[str, np.ndarray]]) -> str:
        if type(doc_summary) is str:
            return doc_summary
        elif type(doc_summary) is np.ndarray:
            return f"Array shaped as {(doc_summary.shape)}: {str(doc_summary.round(4))[:100]}..."
        elif doc_summary is None:
            return str(None)
        else:
            raise NotImplementedError

    @staticmethod
    def build_chunk_id(doc_id: str, position: int) -> str:
        return f"{doc_id}-{position}"

    @property
    def position_in_document(self) -> int:
        chunk_ids = [chunk.chunk_id for chunk in self.belonging_doc.chunks]
        return chunk_ids.index(self.chunk_id)

    def to_json(self) -> ChunkJson:
        return {
            "id": self.chunk_id,
            "text": self.text,
        }

    def chunk_position(self) -> int:
        doc = self.belonging_doc
        chunk_ids = [chunk.chunk_id for chunk in doc.chunks]
        return chunk_ids.index(self.chunk_id)


class DocumentJson(TypedDict):
    id: str
    title: Optional[str]
    chunks: List[ChunkJson]
    candidate_chunk_ids: Optional[List[str]]


@dataclass
class Document:
    """List of chunks."""

    doc_id: str
    chunks: List[Chunk]
    title: Optional[str]
    candidate_chunk_ids: Optional[
        Set[str]
    ] = None  # Which chunks are candidates for retrieval. Mainly for MSMARCO

    def set_default_candidates(self) -> None:
        if self.candidate_chunk_ids is None:
            self.candidate_chunk_ids = {chk.chunk_id for chk in self.chunks}

    @property
    def cid2chunk(self) -> Dict[str, Chunk]:
        """Build a map: From passage ID to the corresponding passage."""
        return {chunk.chunk_id: chunk for chunk in self.chunks}

    @staticmethod
    def build_cid2chunk(corpus: List[Document]) -> Dict[str, Chunk]:
        cid2chunk = {}
        for doc in corpus:
            cid2chunk.update(doc.cid2chunk)
        return cid2chunk

    @staticmethod
    def nchunks_in_corpus(
        corpus: Iterable[Document], candidates_only: bool = False
    ) -> int:
        return sum(
            len(doc.candidate_chunk_ids) if candidates_only else len(doc.chunks)
            for doc in corpus
        )

    @staticmethod
    def nchunks_percentiles(
        corpus: Iterable[Document],
        percentiles: Tuple[int] = (5, 25, 50, 75, 95),
        candidates_only: bool = False,
    ) -> Dict[int, float]:
        nchunks_list = [
            len(doc.candidate_chunk_ids) if candidates_only else len(doc.chunks)
            for doc in corpus
        ]
        result = {
            percentile: np.percentile(nchunks_list, percentile)
            for percentile in percentiles
        }
        return result

    @staticmethod
    def stack_chunks(corpus: List[Document]) -> List[Chunk]:
        """Stack all the chunks in each document all together."""
        return list(chain(*[doc.chunks for doc in corpus]))

    def get_chunk_depth(self, chunk: Chunk) -> int:
        """Return the depth/position of the given passage in the document."""
        for depth, chk in enumerate(self.chunks):
            if chk.chunk_id == chunk.chunk_id:
                return depth
        inf = 10**5  # If not in the document at all
        return inf

    @staticmethod
    def build_did2doc(documents: List[Document]) -> Dict[str, Document]:
        return {doc.doc_id: doc for doc in documents}

    def to_json(self) -> DocumentJson:
        return {
            "id": self.doc_id,
            "title": self.title,
            "chunks": [chunk.to_json() for chunk in self.chunks],
            "candidate_chunk_ids": list(self.candidate_chunk_ids)
            if self.candidate_chunk_ids is not None
            else None,
        }

    @classmethod
    def from_json(cls: Type[Document], doc_json: DocumentJson) -> Document:
        document = cls(doc_id=doc_json["id"], title=doc_json["title"], chunks=[])
        document.chunks = [
            Chunk(
                chunk_id=chk_json["id"],
                text=chk_json["text"],
                doc_summary=None,
                belonging_doc=document,
            )
            for chk_json in doc_json["chunks"]
        ]
        if doc_json.get("candidate_chunk_ids") is not None:
            candidate_chunk_ids = set(doc_json["candidate_chunk_ids"])
            document.candidate_chunk_ids = candidate_chunk_ids
        return document

    @staticmethod
    def split_pool(
        pool: Iterable[Document], batch_size_chunk: int
    ) -> Iterable[List[Document]]:
        batch = []
        naccumulated = 0
        for doc in pool:
            new_size = naccumulated + len(doc.chunks)
            if new_size > batch_size_chunk and len(batch):
                yield batch
                batch = [
                    doc
                ]  # The assertion at the beginning makes sure this < batch size
                naccumulated = len(doc.chunks)
            else:
                batch.append(doc)
                naccumulated = new_size

        # Leftover:
        if len(batch):
            yield batch

    @staticmethod
    def keep_first_chunks_only(pool: Iterable[Document]) -> Iterable[Document]:
        """For FirstP: Keep only the first chunk in each document in the pool."""
        for doc in pool:
            doc_new = Document(
                doc_id=doc.doc_id, chunks=doc.chunks[:1], title=doc.title
            )
            yield doc_new


class QueryJson(TypedDict):
    id: str
    text: str


@dataclass
class Query:
    """A query to be matched (against chunks)."""

    query_id: str
    text: str

    @staticmethod
    def build_qid2query(queries: List[Query]) -> Dict[str, Query]:
        return {q.query_id: q for q in queries}

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.query_id, "text": self.text}


class JudgedChunkJson(TypedDict):
    chunk: ChunkJson
    judgement: int
    belonging_doc: DocumentJson


@dataclass
class JudgedChunk:
    """A chunk labeled with its judgement score."""

    query: Query
    chunk: Chunk
    judgement: int
    model_relevance: Optional[float] = None

    def to_json(self) -> JudgedChunkJson:
        return {
            # "query": self.query.to_json(),
            "chunk": self.chunk.to_json(),
            "judgement": self.judgement,
            "belonging_doc": self.chunk.belonging_doc.to_json(),
        }


class LabeledQueryJson(TypedDict):
    query: QueryJson
    judged_chunks: List[JudgedChunkJson]


@dataclass
class LabeledQuery:
    """Given a query, waht are the judged chunks for it."""

    query: Query
    judged_chunks: List[JudgedChunk]

    @staticmethod
    def njudged_per_query(labeled_queries: List[LabeledQuery]) -> float:
        """How many judged chunks per query."""
        nqueries = LabeledQuery.nqueries(labeled_queries)
        return sum(len(lq.judged_chunks) for lq in labeled_queries) / nqueries

    @staticmethod
    def nqueries(labeled_queries: List[LabeledQuery]) -> int:
        """How many unique queries in these labeled queries."""
        return len({lq.query.query_id for lq in labeled_queries})

    @staticmethod
    def build_cid2judgement(labeled_queries: List[LabeledQuery]) -> Dict[str, int]:
        """Get the map: Chunk ID to judgement for the query."""
        return {
            jchk.chunk.chunk_id: jchk.judgement
            for lq in labeled_queries
            for jchk in lq.judged_chunks
        }

    @staticmethod
    def get_unique_queries(labeled_queries: List[LabeledQuery]) -> List[Query]:
        qid2query = {lq.query.query_id: lq.query for lq in labeled_queries}
        queries = list(qid2query.values())
        return queries

    @staticmethod
    def build_qrels(labeled_queries: List[LabeledQuery]) -> Dict[str, Dict[str, int]]:
        qrels = {}
        for lq in labeled_queries:
            qrels.setdefault(lq.query.query_id, {})
            for jchk in lq.judged_chunks:
                qrels[lq.query.query_id][jchk.chunk.chunk_id] = jchk.judgement
        return qrels

    @staticmethod
    def build_qrels_doc(
        labeled_queries: List[LabeledQuery],
    ) -> Dict[str, Dict[str, int]]:
        qrels = {}
        for lq in labeled_queries:
            qrels.setdefault(lq.query.query_id, {})
            for jchk in lq.judged_chunks:
                qrels[lq.query.query_id][
                    jchk.chunk.belonging_doc.doc_id
                ] = jchk.judgement
        return qrels

    @staticmethod
    def group_by_qid(
        labeled_queries: List[LabeledQuery],
    ) -> Dict[str, List[LabeledQuery]]:
        grouped: Dict[str, List[LabeledQuery]] = {}
        for lq in labeled_queries:
            grouped.setdefault(lq.query.query_id, [])
            grouped[lq.query.query_id].append(lq)
        return grouped

    def to_json(self) -> LabeledQueryJson:
        return {
            "query": self.query.to_json(),
            "judged_chunks": [jchk.to_json() for jchk in self.judged_chunks],
        }

    @staticmethod
    def merge(labeled_queries: List[LabeledQuery]) -> List[LabeledQuery]:
        """Merge labeled queries with the same query ID into one."""
        qid2query = {lq.query.query_id: lq.query for lq in labeled_queries}
        qid2jchks: Dict[str, List[JudgedChunk]] = {}
        for lq in labeled_queries:
            for jchk in lq.judged_chunks:
                qid = jchk.query.query_id
                qid2jchks.setdefault(qid, [])
                qid2jchks[qid].append(jchk)
        lqs = []
        for qid, query in qid2query.items():
            jchks = qid2jchks[qid]
            lq = LabeledQuery(query=query, judged_chunks=jchks)
            lqs.append(lq)
        return lqs


class MetaDataJson(TypedDict):
    chunk_separator: str
    corpus_identifier: str

    name: str
    ndocs: int
    nchunks: int
    nchunks_candidates: Optional[int]
    nchunks_percentiles: Dict[int, float]
    nchunks_candidates_percentiles: Optional[int]
    avg_chunk_length: float
    avg_candidate_chunk_length: Optional[int]
    nqueries_train: Optional[int]
    nqueries_dev: Optional[int]
    nqueries_test: int
    jpq_train: Optional[int]  # Judgements per query
    jpq_dev: Optional[int]
    jpq_test: int


@dataclass
class LoadedData:
    # data:
    corpus_iter_fn: Optional[Callable[[], Iterable[Document]]] = None
    labeled_queries_train: Optional[List[LabeledQuery]] = None
    labeled_queries_dev: Optional[List[LabeledQuery]] = None
    labeled_queries_test: Optional[List[LabeledQuery]] = None

    # meta data:
    meta_data: Optional[MetaDataJson] = None

    def dump(self, output_dir: str) -> None:
        """Dump the data into `corpus.jsonl`, `train/dev/test.jsonl` and `meta_data.json`."""
        os.makedirs(output_dir, exist_ok=True)
        corpus_iter = self.corpus_iter_fn()
        with open(os.path.join(output_dir, "corpus.jsonl"), "w") as f:
            for doc in tqdm.tqdm(
                corpus_iter, total=self.meta_data["ndocs"], desc="Dumping corpus"
            ):
                line = ujson.dumps(doc.to_json()) + "\n"
                f.write(line)
        for lqs, name in zip(
            [
                self.labeled_queries_train,
                self.labeled_queries_dev,
                self.labeled_queries_test,
            ],
            ["train.jsonl", "dev.jsonl", "test.jsonl"],
        ):
            if lqs is None:
                continue
            with open(os.path.join(output_dir, name), "w") as f:
                for lq in tqdm.tqdm(lqs, desc=f"Dumping {name}"):
                    line = ujson.dumps(lq.to_json()) + "\n"
                    f.write(line)

        with open(os.path.join(output_dir, "meta_data.json"), "w") as f:
            f.write(ujson.dumps(self.meta_data, indent=4))

    @staticmethod
    def build_corpus_iter_fn(fpath: str) -> Iterable[Document]:
        with open(fpath) as f:
            for line in f:
                doc_json: DocumentJson = ujson.loads(line)
                document = Document.from_json(doc_json)
                document.set_default_candidates()  # Also for compatibility with the older version
                yield document

    @staticmethod
    def load_labeled_queries(
        fpath: str, total: int, pbar: bool
    ) -> Optional[List[LabeledQuery]]:
        if not os.path.exists(fpath):
            return None

        labeled_queries: List[LabeledQuery] = []
        with open(fpath) as f:
            for line in tqdm.tqdm(
                f, total=total, desc=f"Loading from {fpath}", disable=not pbar
            ):
                lq_json: LabeledQueryJson = ujson.loads(line)
                query_json = lq_json["query"]
                query = Query(query_id=query_json["id"], text=query_json["text"])
                judged_chunk_jsons = lq_json["judged_chunks"]
                judged_chunks = []
                for jchk_json in judged_chunk_jsons:
                    document = Document.from_json(jchk_json["belonging_doc"])
                    chunk_json: ChunkJson = jchk_json["chunk"]
                    chunk = Chunk(
                        chunk_id=chunk_json["id"],
                        text=chunk_json["text"],
                        doc_summary=None,
                        belonging_doc=document,
                    )
                    jchk = JudgedChunk(
                        query=query, chunk=chunk, judgement=jchk_json["judgement"]
                    )
                    judged_chunks.append(jchk)
                labeled_query = LabeledQuery(query=query, judged_chunks=judged_chunks)
                labeled_queries.append(labeled_query)
        return labeled_queries

    @classmethod
    def from_dump(
        cls: Type[LoadedData], dump_dir: str, pbar: bool = True
    ) -> LoadedData:
        with open(os.path.join(dump_dir, "meta_data.json")) as f:
            meta_data: MetaDataJson = ujson.load(f)
            meta_data["chunk_separator"] = Separator(meta_data["chunk_separator"])

        loaded_data = cls(
            corpus_iter_fn=partial(
                cls.build_corpus_iter_fn, os.path.join(dump_dir, "corpus.jsonl")
            ),
            labeled_queries_train=cls.load_labeled_queries(
                fpath=os.path.join(dump_dir, "train.jsonl"),
                total=meta_data["nqueries_train"],
                pbar=pbar,
            ),
            labeled_queries_dev=cls.load_labeled_queries(
                fpath=os.path.join(dump_dir, "dev.jsonl"),
                total=meta_data["nqueries_dev"],
                pbar=pbar,
            ),
            labeled_queries_test=cls.load_labeled_queries(
                fpath=os.path.join(dump_dir, "test.jsonl"),
                total=meta_data["nqueries_test"],
                pbar=pbar,
            ),
            meta_data=meta_data,
        )
        return loaded_data
