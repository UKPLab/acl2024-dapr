from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Union
from dapr.datasets.base import BaseDataset
from dapr.datasets.dm import (
    Chunk,
    Document,
    JudgedChunk,
    LabeledQuery,
    LoadedData,
    Query,
)
from dapr.utils import randomly_split_by_number, set_logger_format
import datasets
import ujson


@dataclass
class DocumentRecord:
    doc_id: str
    url: str
    title: str
    contents: List[str]

    @staticmethod
    def build_url2drecord(drecords: List[DocumentRecord]) -> Dict[str, DocumentRecord]:
        return {drecord.url: drecord for drecord in drecords}


@dataclass
class JudgedDocumentRecord:
    document_record: DocumentRecord
    qid: str
    query: str
    evidence_positions: List[int]


class TaggedConditionalQA(BaseDataset):
    """The original version of ConditionalQA, which contains HTML tags."""

    fdocuments_v1: Optional[str] = None
    ftrain_v1: Optional[str] = None
    fdev_v1: Optional[str] = None

    def _download(self, resource_path: str) -> None:
        """`resource_path` is something like https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0"""
        if os.path.exists(resource_path):
            self.fdocuments_v1 = os.path.join(resource_path, "documents.json")
            self.train_v1 = os.path.join(resource_path, "train.json")
            self.dev_v1 = os.path.join(resource_path, "dev.json")
        else:
            dm = datasets.DownloadManager()
            self.fdocuments_v1 = dm.download(
                os.path.join(resource_path, "documents.json")
            )
            self.ftrain_v1 = dm.download(os.path.join(resource_path, "train.json"))
            self.fdev_v1 = dm.download(os.path.join(resource_path, "dev.json"))

    def _load_drecords(self) -> List[DocumentRecord]:
        with open(self.fdocuments_v1) as f:
            data = ujson.load(f)
        drecords = []
        for i, doc in enumerate(data):
            drecords.append(
                DocumentRecord(
                    doc_id=str(i),
                    url=doc["url"],
                    title=doc["title"],
                    contents=doc["contents"],
                )
            )
        return drecords

    def _load_jrecords(
        self, fpath: str, url2drecod: Dict[str, DocumentRecord]
    ) -> List[JudgedDocumentRecord]:
        with open(fpath) as f:
            data = ujson.load(f)
        jrecords = []
        for example in data:
            if len(example["evidences"]) == 0:
                continue

            drecord = url2drecod[example["url"]]
            query = " ".join([example["scenario"], example["question"]])
            jrecord = JudgedDocumentRecord(
                document_record=drecord,
                qid=example["id"],
                query=query,
                evidence_positions=[
                    drecord.contents.index(evidence)
                    for evidence in example["evidences"]
                ],
            )
            jrecords.append(jrecord)

        return jrecords

    def _build_chunks(
        self, record: Union[JudgedDocumentRecord, DocumentRecord]
    ) -> Union[List[JudgedChunk], List[Chunk]]:
        query: Optional[Query] = None
        drecord: Optional[DocumentRecord] = None
        if type(record) is JudgedDocumentRecord:
            query = Query(query_id=record.qid, text=record.query)
            texts = [evidence for evidence in record.document_record.contents]
            marked = [
                pos in record.evidence_positions
                for pos, _ in enumerate(record.document_record.contents)
            ]
            drecord = record.document_record
        else:
            assert type(record) is DocumentRecord
            texts = [evidence for evidence in record.contents]
            marked = [False] * len(texts)
            drecord = record

        document = Document(doc_id=drecord.doc_id, chunks=[], title=drecord.title)
        judged_chunks = []
        for text, positive in zip(texts, marked):
            chunk_id = Chunk.build_chunk_id(
                doc_id=drecord.doc_id, position=len(document.chunks)
            )
            chunk = Chunk(
                chunk_id=chunk_id, text=text, doc_summary=None, belonging_doc=document
            )
            document.chunks.append(chunk)
            if positive and query is not None:
                judged_chunks.append(JudgedChunk(query=query, chunk=chunk, judgement=1))
        document.set_default_candidates()

        if query is not None:
            return judged_chunks
        else:
            return document.chunks

    def _build_labeled_queries(
        self, jrecords: List[JudgedDocumentRecord]
    ) -> List[LabeledQuery]:
        qid2jchunks: Dict[str, List[JudgedChunk]] = defaultdict(list)
        jchunks_list: List[List[JudgedChunk]] = list(map(self._build_chunks, jrecords))
        for jchunks in jchunks_list:
            for jchunk in jchunks:  # gether jchunks by qid
                qid2jchunks[jchunk.query.query_id].append(jchunk)
        labeled_queries = [
            LabeledQuery(query=jchunks[0].query, judged_chunks=jchunks)
            for jchunks in qid2jchunks.values()
        ]
        return labeled_queries

    def _build_corpus(self, drecords: List[DocumentRecord]) -> List[Document]:
        corpus = []
        chunks_list: List[List[Chunk]] = list(map(self._build_chunks, drecords))
        for chunks in chunks_list:
            doc = chunks[0].belonging_doc
            corpus.append(doc)
        return corpus

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        drecords = self._load_drecords()
        url2drecord = DocumentRecord.build_url2drecord(drecords)
        jrecords_train = self._load_jrecords(
            fpath=self.ftrain_v1, url2drecod=url2drecord
        )
        jrecords_dev = self._load_jrecords(fpath=self.fdev_v1, url2drecod=url2drecord)
        labeled_queries_train_and_dev = self._build_labeled_queries(jrecords_train)
        labeled_queries_test = self._build_labeled_queries(jrecords_dev)
        if nheldout is None:
            nheldout = len(labeled_queries_test)
        labeled_queries_dev, labeled_queries_train = randomly_split_by_number(
            data=labeled_queries_train_and_dev, number=nheldout
        )
        corpus = self._build_corpus(drecords)
        return LoadedData(
            corpus_iter_fn=lambda: iter(corpus),
            labeled_queries_train=labeled_queries_train,
            labeled_queries_dev=labeled_queries_dev,
            labeled_queries_test=labeled_queries_test,
        )


if __name__ == "__main__":
    set_logger_format()
    dataset = TaggedConditionalQA(
        resource_path="https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0",
        nheldout=None,
    )
