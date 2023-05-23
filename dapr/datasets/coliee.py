from collections import defaultdict
from dataclasses import dataclass
import json
import os
import re
import shutil
import tempfile
from typing import Dict, List, Optional, Union
from dapr.datasets.base import BaseDataset, LoadedData
from dapr.datasets.dm import Chunk, Document, JudgedChunk, LabeledQuery, Query
from dapr.utils import concat_and_chunk, set_logger_format
from filelock import FileLock
import gdown
import tqdm


@dataclass
class ParagraphRecord:
    paragraph_id: str
    text: str


@dataclass
class DocRecord:
    doc_id: str
    paragraphs: List[ParagraphRecord]


@dataclass
class JudgedDocRecord:
    query_id: str
    query: str
    doc: DocRecord
    relevant_positions: List[int]


class COLIEE(BaseDataset):
    task1_train_files_2023: Optional[str] = None
    task2_train_files_2023: Optional[str] = None
    task2_train_labels_2023: Optional[str] = None

    def _download(self, resource_path: Optional[str] = None) -> None:
        if os.path.exists(resource_path):
            self.task1_train_files_2023 = os.path.join(
                resource_path, "task1_train_files_2023"
            )
            self.task2_train_files_2023 = os.path.join(
                resource_path, "task2_train_files_2023"
            )
            self.task2_train_labels_2023 = os.path.join(
                resource_path, "task2_train_labels_2023.json"
            )
        else:
            try:
                task1_train_files_2023 = os.environ["COLIEE_TASK1_TRAIN_FILES"]
                task2_train_files_2023 = os.environ["COLIEE_TASK2_TRAIN_FILES"]
                task2_train_labels_2023 = os.environ["COLIEE_TASK2_TRAIN_LABELS"]
            except KeyError as e:
                self.logger.error(
                    "Please set up the Google drive IDs of the COLIEE data in the environment variables:\n"
                    "COLIEE_TASK1_TRAIN_FILES\nCOLIEE_TASK2_TRAIN_FILES\nCOLIEE_TASK2_TRAIN_LABELS"
                )
                raise e

            tmp_dir = tempfile.gettempdir()
            self.task1_train_files_2023 = os.path.join(
                tmp_dir, "task1_train_files_2023"
            )
            self.task2_train_files_2023 = os.path.join(
                tmp_dir, "task2_train_files_2023"
            )
            self.task2_train_labels_2023 = os.path.join(
                tmp_dir, "task2_train_labels_2023.json"
            )
            with FileLock(self.task1_train_files_2023 + ".lock"):
                if not os.path.exists(self.task1_train_files_2023):
                    gdown.download(
                        id=task1_train_files_2023,
                        output=self.task1_train_files_2023 + ".zip",
                    )
                    gdown.extractall(path=self.task1_train_files_2023 + ".zip")
                    # shutil.rmtree(os.path.join(self.task1_train_files_2023, "__MACOSX"))
            with FileLock(self.task2_train_files_2023 + ".lock"):
                if not os.path.exists(self.task2_train_files_2023):
                    gdown.download(
                        id=task2_train_files_2023,
                        output=self.task2_train_files_2023 + ".zip",
                    )
                    gdown.extractall(path=self.task2_train_files_2023 + ".zip")
                    redundant = os.path.join(
                        self.task2_train_files_2023, "task2_train_labels_2022.json"
                    )
                    if os.path.exists(redundant):
                        os.remove(redundant)
                        self.logger.info(f"Removed redundant file: {redundant}")
                    # shutil.rmtree(os.path.join(self.task2_train_files_2023, "__MACOSX"))
            with FileLock(self.task2_train_labels_2023 + ".lock"):
                if not os.path.exists(self.task2_train_labels_2023):
                    gdown.download(
                        id=task2_train_labels_2023,
                        output=self.task2_train_labels_2023,
                    )

    def _build_drecords_task1(self) -> List[DocRecord]:
        drecords: List[DocRecord] = []
        for doc in tqdm.tqdm(
            sorted(os.listdir(self.task1_train_files_2023)),
            desc="Building task1 records",
        ):
            # doc: 000028.txt
            doc_id = doc.replace(".txt", "")
            drecord = DocRecord(doc_id=doc_id, paragraphs=[])
            doc_path = os.path.join(self.task1_train_files_2023, doc)
            with open(doc_path) as f:
                text = f.read().strip()
            matches: List[re.Match] = list(
                re.finditer("\[\s*\d+\s*\]", text)
            )  # Get a paragraph between each two ([number], [number])'s
            assert len(matches) >= 1
            starts = [m.start() for m in matches]
            for i, (b, e) in enumerate(zip(starts, starts[1:] + [len(text)])):
                paragraph = text[b:e]
                drecord.paragraphs.append(
                    ParagraphRecord(paragraph_id=str(i), text=paragraph)
                )
            drecords.append(drecord)
        return drecords

    def _build_drecords_task2(self) -> List[DocRecord]:
        drecords: List[DocRecord] = []
        for doc_folder in tqdm.tqdm(
            sorted(os.listdir(self.task2_train_files_2023)),
            desc="Building task2 records",
        ):
            # doc_folder: 001
            drecord = DocRecord(doc_id=doc_folder, paragraphs=[])
            paragraphs_folder = os.path.join(
                self.task2_train_files_2023, doc_folder, "paragraphs"
            )
            for paragraph in sorted(os.listdir(paragraphs_folder)):
                # paragraph: 001.txt
                with open(os.path.join(paragraphs_folder, paragraph)) as f:
                    ptext = f.read().strip()
                    precord = ParagraphRecord(paragraph_id=f"{paragraph}", text=ptext)
                    drecord.paragraphs.append(precord)
            drecords.append(drecord)
        return drecords

    def _build_jrecords(self, drecords_task2: List[DocRecord]) -> List[JudgedDocRecord]:
        doc_id2drecord_task2 = {doc.doc_id: doc for doc in drecords_task2}
        with open(self.task2_train_labels_2023) as f:
            labels: Dict[str, List[str]] = json.load(f)
            # {
            # "001": [
            #     "027.txt"
            # ],
            # "002": [
            #     "014.txt"
            # ],
            # "003": [
            #     "003.txt",
            #     "004.txt"
            # ],
            # ...
            # }
        jdrecords: List[JudgedDocRecord] = []
        for doc_folder in tqdm.tqdm(
            sorted(os.listdir(self.task2_train_files_2023)),
            desc="Building judged records",
        ):
            query_path = os.path.join(
                self.task2_train_files_2023, doc_folder, "entailed_fragment.txt"
            )
            with open(query_path) as f:
                query = f.read().strip()
            qid = doc_id = doc_folder
            doc = doc_id2drecord_task2[doc_id]
            jdrecord = JudgedDocRecord(
                query_id=qid, query=query, doc=doc, relevant_positions=[]
            )
            pid2relevant_position = {
                precord.paragraph_id: pos for pos, precord in enumerate(doc.paragraphs)
            }
            for pid in labels[qid]:
                position = pid2relevant_position[pid]
                jdrecord.relevant_positions.append(position)
            jdrecords.append(jdrecord)
        return jdrecords

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
        document = Document(doc_id=drecord.doc_id, chunks=[], title=None)
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

        if query is not None:
            return judged_chunks
        else:
            return document.chunks

    def _build_labeled_queries(
        self, jrecords: List[JudgedDocRecord]
    ) -> List[LabeledQuery]:
        qid2jchunks: Dict[str, List[JudgedChunk]] = defaultdict(list)
        for jrecord in tqdm.tqdm(jrecords, desc="Building labeled queries"):
            jchunks = self._build_chunks(jrecord)
            for jchunk in jchunks:  # gether jchunks by qid
                qid2jchunks[jchunk.query.query_id].append(jchunk)
        labeled_queries = [
            LabeledQuery(query=jchunks[0].query, judged_chunks=jchunks)
            for jchunks in qid2jchunks.values()
        ]
        return labeled_queries

    def _build_corpus(self, drecords: List[DocRecord]) -> List[Document]:
        corpus = []
        for drecord in tqdm.tqdm(drecords, desc="Building corpus"):
            chunks: List[Chunk] = self._build_chunks(drecord)
            doc = chunks[0].belonging_doc
            corpus.append(doc)
        return corpus

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        drecords_task1 = self._build_drecords_task1()
        drecords_task2 = self._build_drecords_task2()
        jrecords = self._build_jrecords(drecords_task2)
        labeled_queries = self._build_labeled_queries(jrecords)
        corpus = self._build_corpus(drecords_task1 + drecords_task2)
        return LoadedData(
            corpus_iter_fn=lambda: iter(corpus),
            labeled_queries_test=labeled_queries,
        )


if __name__ == "__main__":
    import crash_ipdb

    set_logger_format()
    dataset = COLIEE(
        resource_path="", nheldout=None, cache_loaded=False, max_nchunks=None
    )
    # [2023-03-12 20:40:58] INFO [dadpr.datasets.base.stats:161]
    # {
    #     "name": "COLIEE",
    #     "#docs": 5025,
    #     "#chks": 238709,
    #     "#chks percentiles": {
    #         "5": 12.0,
    #         "25": 24.0,
    #         "50": 37.0,
    #         "75": 55.0,
    #         "95": 111.0
    #     },
    #     "avg. chunk length": 145.55,
    #     "#train queries": null,
    #     "#Judged per query (train)": null,
    #     "#dev queries": null,
    #     "#Judged per query (dev)": null,
    #     "#test queries": 625,
    #     "#Judged per query (test)": 1.17
    # }
