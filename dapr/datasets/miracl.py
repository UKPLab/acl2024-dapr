from collections import defaultdict
from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Union
from dapr.datasets.base import BaseDataset, LoadedData
from dapr.datasets.dm import Chunk, Document, JudgedChunk, LabeledQuery, Query
import datasets
from dapr.utils import (
    Multiprocesser,
    Separator,
    randomly_split_by_number,
    set_logger_format,
    tqdm_ropen,
)
import tqdm


@dataclass
class ParagraphRecord:
    paragraph_id: str
    text: str


@dataclass
class DocRecord:
    doc_id: str
    title: str
    paragraphs: List[ParagraphRecord]


@dataclass
class JudgedDocRecord:
    query_id: str
    query: str
    doc: DocRecord
    relevant_positions: List[int]


class MIRACL(BaseDataset):
    corpus_files: Optional[List[str]] = None
    fqrels_train: Optional[str] = None
    fqrels_dev: Optional[str] = None
    ftopics_train: Optional[str] = None
    ftopics_dev: Optional[str] = None

    def __init__(
        self,
        resource_path: str = "https://huggingface.co/datasets/miracl",
        nheldout: Optional[int] = None,
        cache_root_dir: str = "data",
        chunk_separator: Separator = Separator.empty,
        tokenizer: str = "roberta-base",
        nprocs: int = 10,
    ) -> None:
        super().__init__(
            resource_path, nheldout, cache_root_dir, chunk_separator, tokenizer, nprocs
        )

    def _download(self, resource_path: str) -> None:
        """The resource path should be something like https://huggingface.co/datasets/miracl."""
        corpus_splits = [f"docs-{i}.jsonl" for i in range(66)]
        if os.path.exists(resource_path):
            self.corpus_files = [
                os.path.join(
                    resource_path, "miracl-corpus", "miracl-corpus-v1.0-en", split
                )
                for split in corpus_splits
            ]
            self.fqrels_dev = os.path.join(
                resource_path,
                "miracl",
                "miracl-v1.0-en",
                "qrels",
                "qrels.miracl-v1.0-en-dev.tsv",
            )
            self.fqrels_train = os.path.join(
                resource_path,
                "miracl",
                "miracl-v1.0-en",
                "qrels",
                "qrels.miracl-v1.0-en-train.tsv",
            )
            self.ftopics_dev = os.path.join(
                resource_path,
                "miracl",
                "miracl-v1.0-en",
                "topics",
                "topics.miracl-v1.0-en-dev.tsv",
            )
            self.ftopics_train = os.path.join(
                resource_path,
                "miracl",
                "miracl-v1.0-en",
                "topics",
                "topics.miracl-v1.0-en-train.tsv",
            )
        else:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = (
                "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"  # One needs set up the HF token https://huggingface.co/settings/tokens
            )
            dm_config = datasets.DownloadConfig(use_auth_token=True)
            dm = datasets.DownloadManager(download_config=dm_config)
            self.corpus_files = [
                dm.download_and_extract(
                    os.path.join(
                        resource_path,
                        "miracl-corpus",
                        "resolve",
                        "main",
                        "miracl-corpus-v1.0-en",
                        f"{split}.gz",
                    )
                )
                for split in corpus_splits
            ]
            self.fqrels_dev = dm.download(
                os.path.join(
                    resource_path,
                    "miracl",
                    "raw",
                    "main",
                    "miracl-v1.0-en",
                    "qrels",
                    "qrels.miracl-v1.0-en-dev.tsv",
                )
            )
            self.fqrels_train = dm.download(
                os.path.join(
                    resource_path,
                    "miracl",
                    "raw",
                    "main",
                    "miracl-v1.0-en",
                    "qrels",
                    "qrels.miracl-v1.0-en-train.tsv",
                )
            )
            self.ftopics_dev = dm.download(
                os.path.join(
                    resource_path,
                    "miracl",
                    "raw",
                    "main",
                    "miracl-v1.0-en",
                    "topics",
                    "topics.miracl-v1.0-en-dev.tsv",
                )
            )
            self.ftopics_train = dm.download(
                os.path.join(
                    resource_path,
                    "miracl",
                    "raw",
                    "main",
                    "miracl-v1.0-en",
                    "topics",
                    "topics.miracl-v1.0-en-train.tsv",
                )
            )

    def _build_drecords(self) -> List[DocRecord]:
        lines = []
        for corpus_file in self.corpus_files:
            for line in tqdm_ropen(
                fpath=corpus_file, desc=f"Loading corpus file {corpus_file}"
            ):
                lines.append(line)
        line_dicts = Multiprocesser(self.nprocs).run(
            data=lines, func=json.loads, desc="Parsing jsonl", total=len(lines)
        )
        did2precords: Dict[str, List[ParagraphRecord]] = {}
        did2title: Dict[str, str] = {}
        for line_dict in line_dicts:
            pid: str = line_dict["docid"]  # something like "39#0"
            text = line_dict["text"]
            title = line_dict["title"]
            did, position = pid.split("#")
            did2title[did] = title
            precord = ParagraphRecord(paragraph_id=position, text=text)
            did2precords.setdefault(did, list())
            did2precords[did].append(precord)
        drecords = [
            DocRecord(
                doc_id=did,
                title=did2title[did],
                # NOTICE: the sorting key should be integers but strings!!!
                paragraphs=sorted(
                    precords, key=lambda precord: int(precord.paragraph_id)
                ),
            )
            for did, precords in tqdm.tqdm(
                did2precords.items(), desc="Organizing document records"
            )
        ]
        return drecords

    def _build_jrecords(
        self, drecords: List[DocRecord], fqrels: str, ftopics: str
    ) -> List[JudgedDocRecord]:
        qid2pids: Dict[str, List[str]] = {}
        for line in tqdm_ropen(fpath=fqrels, desc=f"Loading qrels from {fqrels}"):
            #  0	Q0	462221#4	1
            qid, _, pid, label = line.strip().split("\t")
            if int(label):
                qid2pids.setdefault(qid, list())
                qid2pids[qid].append(pid)
        qid2query: Dict[str, str] = {}
        for line in tqdm_ropen(fpath=ftopics, desc=f"Loading topics from {ftopics}"):
            qid, query = line.strip().split("\t")  # 0	Is Creole a pidgin of French?
            qid2query[qid] = query
        jrecords: List[JudgedDocRecord] = []
        did2drecord = {drecord.doc_id: drecord for drecord in drecords}
        for qid, pids in tqdm.tqdm(qid2pids.items(), desc="Building jrecords"):
            did2positions: Dict[str, List[int]] = {}
            for pid in pids:
                did, position = pid.split("#")
                did2positions.setdefault(did, list())
                did2positions[did].append(int(position))
            for did, positions in did2positions.items():
                jrecord = JudgedDocRecord(
                    query_id=qid,
                    query=qid2query[qid],
                    doc=did2drecord[did],
                    relevant_positions=positions,
                )
                jrecords.append(jrecord)
        return jrecords

    def _build_chunks(
        self, record: Union[JudgedDocRecord, DocRecord]
    ) -> Union[List[JudgedChunk], List[Chunk]]:
        query: Optional[Query] = None
        drecord: Optional[DocRecord] = None
        if type(record) is JudgedDocRecord:
            query = Query(query_id=record.query_id, text=record.query)
            texts = [precord.text for precord in record.doc.paragraphs]
            marked = [
                pos in record.relevant_positions
                for pos, _ in enumerate(record.doc.paragraphs)
            ]
            drecord = record.doc
        else:
            assert type(record) is DocRecord
            texts = [precord.text for precord in record.paragraphs]
            marked = [False] * len(texts)
            drecord = record

        input_ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        document = Document(doc_id=drecord.doc_id, chunks=[], title=drecord.title)
        judged_chunks = []
        for chunk_token_ids, positive in zip(input_ids, marked):
            chunk_id = Chunk.build_chunk_id(
                doc_id=drecord.doc_id, position=len(document.chunks)
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                text=self.tokenizer.decode(chunk_token_ids),
                doc_summary=None,
                belonging_doc=document,
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
        self, jrecords: List[JudgedDocRecord]
    ) -> List[LabeledQuery]:
        qid2jchunks: Dict[str, List[JudgedChunk]] = defaultdict(list)
        jchunks_list: List[List[JudgedChunk]] = Multiprocesser(self.nprocs).run(
            data=jrecords,
            func=self._build_chunks,
            desc="Building labeled queries",
            total=len(jrecords),
        )
        for jchunks in jchunks_list:
            for jchunk in jchunks:  # gether jchunks by qid
                qid2jchunks[jchunk.query.query_id].append(jchunk)
        labeled_queries = [
            LabeledQuery(query=jchunks[0].query, judged_chunks=jchunks)
            for jchunks in qid2jchunks.values()
        ]
        return labeled_queries

    def _build_corpus(self, drecords: List[DocRecord]) -> List[Document]:
        corpus = []
        chunks_list: List[List[Chunk]] = Multiprocesser(self.nprocs).run(
            data=drecords,
            func=self._build_chunks,
            desc="Building corpus",
            total=len(drecords),
        )
        for chunks in chunks_list:
            doc = chunks[0].belonging_doc
            corpus.append(doc)
        return corpus

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        drecords = self._build_drecords()
        jrecords_train = self._build_jrecords(
            drecords=drecords, fqrels=self.fqrels_train, ftopics=self.ftopics_train
        )
        jrecords_dev = self._build_jrecords(
            drecords=drecords, fqrels=self.fqrels_dev, ftopics=self.ftopics_dev
        )
        labeled_queries_test = self._build_labeled_queries(jrecords_dev)
        labeled_queries_train_and_dev = self._build_labeled_queries(jrecords_train)
        nheldout = len(labeled_queries_test) if nheldout is None else nheldout
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
    from dapr.utils import set_logger_format

    set_logger_format()
    dataset = MIRACL()
