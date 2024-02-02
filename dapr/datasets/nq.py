from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json
import os
from typing import Iterator, List, Optional, Set, Union

import tqdm
from dapr.utils import (
    Multiprocesser,
    Separator,
    tqdm_ropen,
    randomly_split_by_number,
)
from dapr.datasets.dm import (
    Document,
    JudgedChunk,
    LabeledQuery,
    Chunk,
    Query,
    Split,
)
from dapr.datasets.base import BaseDataset, LoadedData
from datasets import DownloadManager


class PassageType(str, Enum):
    """Passage types in NQ."""

    text = "text"
    table = "table"
    list = "list"
    list_definition = "list_definition"
    invalid = "invalid"


@dataclass
class QuestionRecord:
    """A question record in NQ."""

    question: str
    title: str
    candidates: List[str]
    long_answers: List[int]
    passage_types: List[PassageType]

    record_id: (
        str  # Can works as the query ID or the document ID: train/dev + line number
    )

    def get_candidate_ids(self) -> List[str]:
        """Unique identifier for a candidate. Can work as a passage ID."""
        return [
            f"{self.record_id}-{position}" for position in range(len(self.candidates))
        ]

    def get_long_answer_types(self) -> Iterator[PassageType]:
        """Get all the type info of each long-answer candidate."""
        return map(
            lambda long_answer: self.passage_types[long_answer],
            self.long_answers,
        )

    def get_text_candidate_positions(self) -> Iterator[str]:
        """Get all the text candidate positions"""
        return filter(
            lambda position: self.passage_types[position] is PassageType.text,
            range(len(self.candidates)),
        )

    def get_positive_candidate_positions(self) -> Iterator[int]:
        """Get all the positive candidiate positions."""
        positive_positions = filter(
            lambda long_answer: self.passage_types[long_answer] is PassageType.text,
            self.long_answers,
        )
        return positive_positions

    def get_negative_candidate_positions(self) -> Iterator[int]:
        """Get all the negative candidiate positions."""
        negative_positions = filter(
            lambda candidate_position: self.passage_types[candidate_position]
            is PassageType.text
            and candidate_position not in self.long_answers,
            range(len(self.candidates)),
        )
        return negative_positions

    @classmethod
    def from_line(cls, line: str, split: Split, line_number: int) -> QuestionRecord:
        """Instantiate a question record from a dumped json line."""
        line_dict = json.loads(line)
        kwargs = {}
        for arg in QuestionRecord.__dataclass_fields__:
            if arg not in line_dict:
                continue

            value = line_dict[arg]
            if arg == "passage_types":
                value = list(map(PassageType, line_dict[arg]))
            kwargs[arg] = value

        kwargs["record_id"] = f"{split}-{line_number}"
        return cls(**kwargs)

    def has_text_long_answer(self) -> bool:
        """Whether this question record a text-typed long answer."""
        return any(
            map(
                lambda position: len(self.candidates[position]) > 0,
                self.get_positive_candidate_positions(),
            )
        )

    def has_in_doc_negative(self) -> bool:
        """Whether the question record has at least one in-document negative"""
        positive_positions = set(self.get_positive_candidate_positions())
        negative_positions = filter(
            lambda position: position not in positive_positions,
            self.get_text_candidate_positions(),
        )
        return any(
            map(lambda position: len(self.candidates[position]) > 0, negative_positions)
        )

    @staticmethod
    def uniquely_titled(qrecords: Iterator[QuestionRecord]) -> Iterator[QuestionRecord]:
        """Take the first question record with the first occurrence of its title."""
        titles: Set[str] = set()
        for qrecord in qrecords:
            title = qrecord.title.lower()
            if title in titles:
                continue

            yield qrecord
            titles.add(title)


class NaturalQuestions(BaseDataset):
    """NaturalQuestions: Load the dataset into corpus and labeled queries."""

    ftrain: Optional[str] = None
    fdev: Optional[str] = None

    def __init__(
        self,
        resource_path: str = "https://huggingface.co/datasets/sentence-transformers/NQ-retrieval",
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
        """The resource path should be something like https://huggingface.co/datasets/sentence-transformers/NQ-retrieval."""
        if os.path.exists(resource_path):
            self.fdev = os.path.join(resource_path, "dev.jsonl")
            self.ftrain = os.path.join(resource_path, "train.jsonl")
        else:
            fdev = os.path.join(resource_path, "resolve", "main", "dev.jsonl.gz")
            ftrain = os.path.join(resource_path, "resolve", "main", "train.jsonl.gz")
            dm = DownloadManager()
            self.fdev = dm.download_and_extract(fdev)
            self.ftrain = dm.download_and_extract(ftrain)

    def _read_and_clean(self, fpath: str, split: Split) -> List[QuestionRecord]:
        """Read question records, while skipping empty questions and questions without a text answer."""
        qrecords: List[QuestionRecord] = []
        for line_number, line in enumerate(
            tqdm_ropen(fpath, f"Reading records from {os.path.basename(fpath)}")
        ):
            qrecord = QuestionRecord.from_line(line, split, line_number)
            doc_length = sum(
                map(
                    lambda position: len(qrecord.candidates[position]),
                    qrecord.get_text_candidate_positions(),
                )
            )
            if doc_length == 0:
                continue

            if split is split.dev and not qrecord.has_text_long_answer():
                # No need to clean train split!!!
                continue
            qrecords.append(qrecord)
        return qrecords

    def _build_chunks(
        self, qrecord: QuestionRecord, judged: bool
    ) -> Union[List[JudgedChunk], List[Chunk]]:
        query = Query(query_id=qrecord.record_id, text=qrecord.question)
        passages = list(
            map(
                lambda position: qrecord.candidates[position],
                qrecord.get_text_candidate_positions(),
            )
        )
        positive_posistions = set(qrecord.get_positive_candidate_positions())
        text_positions = list(qrecord.get_text_candidate_positions())
        marked = map(lambda pos: pos in positive_posistions, text_positions)
        input_ids = self.tokenizer(passages, add_special_tokens=False)["input_ids"]
        judged_chunks = []
        document = Document(doc_id=qrecord.record_id, chunks=[], title=qrecord.title)
        for chunk_token_ids, positive in zip(input_ids, marked):
            chunk_id = Chunk.build_chunk_id(
                doc_id=document.doc_id, position=len(document.chunks)
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                text=self.tokenizer.decode(chunk_token_ids),
                doc_summary=None,
                belonging_doc=document,
            )
            document.chunks.append(chunk)
            if positive:
                judged_chunks.append(JudgedChunk(query=query, chunk=chunk, judgement=1))
        document.set_default_candidates()

        if judged:
            return judged_chunks
        else:
            return document.chunks

    def _build_document(self, qrecord: QuestionRecord) -> Document:
        chunks: List[Chunk] = self._build_chunks(qrecord=qrecord, judged=False)
        return chunks[0].belonging_doc

    def _build_corpus(self, qrecords: Iterator[QuestionRecord]) -> List[Document]:
        """Build corpus i.e. list of documents from question records. Record ID will be used as document IDs."""
        qrecords = list(qrecords)
        corpus = Multiprocesser(self.nprocs).run(
            data=qrecords,
            func=self._build_document,
            desc="Building corpus",
            total=len(qrecords),
        )
        return corpus

    def _build_labeled_queries(
        self, qrecords: Iterator[QuestionRecord]
    ) -> List[LabeledQuery]:
        """Build the labeled queries from question records. Record ID will be used as both query and document IDs."""
        labeled_queries = []
        for qrecord in tqdm.tqdm(list(qrecords), desc="Building labeled queries"):
            judged_chunks: List[JudgedChunk] = self._build_chunks(
                qrecord=qrecord, judged=True
            )
            query = judged_chunks[0].query
            labeled_queries.append(
                LabeledQuery(
                    query=query,
                    judged_chunks=judged_chunks,
                )
            )
        return labeled_queries

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        qrecords_dev = self._read_and_clean(self.fdev, Split.dev)
        qrecords_train = self._read_and_clean(self.ftrain, Split.train)

        # Build the corpus:
        corpus = self._build_corpus(
            QuestionRecord.uniquely_titled(qrecords_dev + qrecords_train)
        )
        record_ids_in_corpus = set(
            [doc.doc_id for doc in corpus]
        )  # doc_id is equal to record_id

        # Build the labeled queries:
        labeled_queries_test = self._build_labeled_queries(
            filter(
                lambda qrecord: qrecord.record_id in record_ids_in_corpus, qrecords_dev
            )
        )
        # Split the original train split into our dev and train splits
        nheldout = len(labeled_queries_test) if nheldout is None else nheldout
        qrecords_heldout: List[QuestionRecord] = randomly_split_by_number(
            list(
                filter(
                    lambda qrecord: qrecord.record_id in record_ids_in_corpus
                    and qrecord.has_text_long_answer(),
                    qrecords_train,
                )
            ),
            nheldout,
        )[0]
        labeled_queries_dev = self._build_labeled_queries(qrecords_heldout)

        # Use the rest of records for training (and there could be duplicate titles)
        # There should be no intersection interms of titles, too
        # Each data point should have at least one positive and one in-doc negative
        record_ids_heldout = set([qrecord.record_id for qrecord in qrecords_heldout])
        titles_heldout = set([qrecord.title.lower() for qrecord in qrecords_heldout])
        labeled_queries_train = self._build_labeled_queries(
            filter(
                lambda qrecord: qrecord.record_id not in record_ids_heldout
                and qrecord.has_text_long_answer()
                and qrecord.has_in_doc_negative()
                and qrecord.title.lower() not in titles_heldout,
                qrecords_train,
            )
        )

        return LoadedData(
            corpus_iter_fn=lambda: iter(corpus),
            labeled_queries_train=labeled_queries_train,
            labeled_queries_dev=labeled_queries_dev,
            labeled_queries_test=labeled_queries_test,
        )


if __name__ == "__main__":
    from dapr.utils import set_logger_format

    set_logger_format()
    nq = NaturalQuestions()
