from collections import defaultdict
from dataclasses import dataclass, replace
import os
import random
from typing import Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch

import datasets
from dapr.datasets.base import BaseDataset, LoadedData
from dapr.datasets.dm import Chunk, Document, JudgedChunk, LabeledQuery, Query
from dapr.utils import (
    concat_and_chunk,
    download,
    randomly_split_by_number,
    tqdm_ropen,
    Multiprocesser,
    set_logger_format,
)
from datasets import DownloadManager
import pandas as pd
import regex
import tqdm


@dataclass
class DocumentRecord:
    doc_id: str
    title: str
    text: str
    url: str


@dataclass
class JudgedDocumentRecord:
    query: str
    doc_id: str
    title: str
    text_split: List[str]  # Should be equal to DocumentRecord.text after concatenated
    judgements: List[bool]


@dataclass
class PassageRecord:
    query_id: str
    query: str
    # is_selected: bool  # Keep only positive
    url: str
    passage_text: str
    span: Optional[Tuple[int, int]] = None


class MSMARCO(BaseDataset):
    """MSMARCO: Load the dataset into corpus and labeled queries."""

    fcorpus: Optional[str] = None
    ftrain: Optional[str] = None
    fdev: Optional[str] = None
    max_mismatch: int = 5

    def _download(self, resource_path: str) -> None:
        """The resource path should be something like https://msmarco.blob.core.windows.net."""
        if os.path.exists(resource_path):
            self.fcorpus = os.path.join(resource_path, "msmarco-docs.tsv")
            self.ftrain = os.path.join(resource_path, "train_v2.1.json")
            self.fdev = os.path.join(resource_path, "dev_v2.1.json")
        else:
            fcorpus = os.path.join(resource_path, "msmarcoranking/msmarco-docs.tsv.gz")
            ftrain = os.path.join(resource_path, "msmarco/train_v2.1.json.gz")
            fdev = os.path.join(resource_path, "msmarco/dev_v2.1.json.gz")
            with patch.object(datasets.utils.file_utils, "http_get", download):
                dm = DownloadManager()
                self.fcorpus = dm.download_and_extract(fcorpus)
                self.ftrain = dm.download_and_extract(ftrain)
                self.fdev = dm.download_and_extract(fdev)

    def _build_document_records(self) -> List[DocumentRecord]:
        records = []
        for line in tqdm_ropen(
            self.fcorpus, f"Reading corpus file {os.path.basename(self.fcorpus)}"
        ):
            doc_id, url, title, text = line.split("\t")
            text = text.strip()
            if len(text) == 0:
                continue

            records.append(
                DocumentRecord(url=url, title=title, text=text, doc_id=doc_id)
            )
        return records

    def _read_passage_records(self, fpath: str) -> List[PassageRecord]:
        self.logger.info(f"Reading QNA records from {os.path.basename(fpath)}")
        df = pd.read_json(fpath)
        records = []
        for _, line_dict in df.iterrows():
            for psg in line_dict["passages"]:
                if psg["is_selected"]:
                    record = PassageRecord(
                        query_id=str(line_dict["query_id"]),
                        query=line_dict["query"],
                        url=psg["url"],
                        passage_text=psg["passage_text"],
                    )
                    records.append(record)
        return records

    def _filter_passage_records(
        self, passage_records: List[PassageRecord], urls: Set[str]
    ) -> List[PassageRecord]:
        """Filter the QNA records to make sure all the passage urls are in the given url set."""
        records = list(
            filter(
                lambda psg: psg.url in urls,
                tqdm.tqdm(passage_records, desc="Filtering records"),
            )
        )
        return records

    def _find_match(self, precord_and_doc: Tuple[PassageRecord, str]) -> PassageRecord:
        precord, doc = precord_and_doc
        new_record = replace(precord)
        psg_text = new_record.passage_text
        found = doc.find(psg_text)
        if found != -1:
            new_record.span = (found, found + len(psg_text))
        else:
            pattern = "(%s){e<=%d}" % (
                regex.escape(psg_text),
                self.max_mismatch,
            )
            match = regex.compile(pattern).search(doc)
            if match is not None:
                new_record.span = match.span()
        return new_record

    def _find_matches(
        self,
        precords: List[PassageRecord],
        drecords: List[DocumentRecord],
    ) -> List[PassageRecord]:
        url2doc = {r.url: r.text for r in drecords}
        docs = [url2doc[r.url] for r in precords]
        results = Multiprocesser(self.nprocs).run(
            data=zip(precords, docs),
            func=self._find_match,
            desc="Finding matches",
            total=len(precords),
        )
        return results

    def _build_chunks(
        self,
        drecord_and_precord: Tuple[DocumentRecord, Optional[PassageRecord]],
    ) -> Union[List[JudgedChunk], List[Chunk]]:
        drecord, precord = drecord_and_precord
        query: Optional[Query] = None

        # Build the spans:
        texts: List[str]
        marked: List[bool]
        if precord is not None:
            query = Query(query_id=precord.query_id, text=precord.query)
            assert precord.span is not None
            b, e = precord.span
            texts = [drecord.text[:b], drecord.text[b:e], drecord.text[e:]]
            marked = [False, True, False]
            texts, marked = zip(
                *filter(lambda text_marked: len(text_marked[0]), zip(texts, marked))
            )
            texts = list(texts)
        else:
            texts = [drecord.text]
            marked = [False]

        # Build the chunks:
        input_ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        document = Document(doc_id=drecord.doc_id, chunks=[], title=drecord.title)
        judged_chunks = []
        for chunk_token_ids, positive in concat_and_chunk(
            sequences=input_ids, marked=marked, chunk_size=self.chunk_size
        ):
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

        if precord is None:
            return document.chunks
        else:
            return judged_chunks

    def _build_document(self, drecord: DocumentRecord) -> Document:
        chunks: List[Chunk] = self._build_chunks((drecord, None))
        return chunks[0].belonging_doc

    def _build_corpus(self, drecords: List[DocumentRecord]) -> List[Document]:
        corpus = []
        corpus = Multiprocesser(self.nprocs).run(
            data=drecords,
            func=self._build_document,
            desc="Building corpus",
            total=len(drecords),
        )
        return corpus

    def _build_labeled_queries(
        self,
        passage_records: List[PassageRecord],
        url2drecord: Dict[str, DocumentRecord],
    ) -> List[LabeledQuery]:
        drecords = [url2drecord[precord.url] for precord in passage_records]
        drecords_and_precords = list(
            filter(
                lambda dprecords: len(dprecords[0].text)
                and dprecords[1].span is not None,
                zip(drecords, passage_records),
            )
        )
        jchunks_list = Multiprocesser(self.nprocs).run(
            data=drecords_and_precords,
            func=self._build_chunks,
            desc="Building labeled queries",
            total=len(drecords_and_precords),
        )
        qid2jchunks: Dict[str, List[JudgedChunk]] = defaultdict(list)
        for jchunks in jchunks_list:
            for jchunk in jchunks:
                qid2jchunks[jchunk.query.query_id].append(jchunk)
        labeled_queries = [
            LabeledQuery(query=jchunks[0].query, judged_chunks=jchunks)
            for jchunks in qid2jchunks.values()
        ]
        return labeled_queries

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        # Load and filter data:
        drecords = self._build_document_records()
        url2drecord = {record.url: record for record in drecords}
        urls = set(url2drecord)

        dev_records = self._find_matches(
            precords=self._filter_passage_records(
                passage_records=self._read_passage_records(self.fdev), urls=urls
            ),
            drecords=drecords,
        )

        train_records = self._find_matches(
            precords=self._filter_passage_records(
                passage_records=self._read_passage_records(self.ftrain), urls=urls
            ),
            drecords=drecords,
        )

        # Build corpus and labeled queries:
        labeled_queries_dev_and_test = self._build_labeled_queries(
            passage_records=dev_records, url2drecord=url2drecord
        )
        nheldout = (
            len(labeled_queries_dev_and_test) // 2 if nheldout is None else nheldout
        )
        labeled_queries_train = self._build_labeled_queries(
            passage_records=train_records, url2drecord=url2drecord
        )
        labeled_queries_dev, labeled_queries_test = randomly_split_by_number(
            data=labeled_queries_dev_and_test, number=nheldout
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
    msmarco = MSMARCO(
        resource_path="https://msmarco.blob.core.windows.net",
        # resource_path="/home/fb20user07/research/msmarco/all",
        nheldout=None,
        cache_loaded=True,
    )

    output_dir = os.path.join("sample-data", msmarco.name)
    if not os.path.exists(output_dir):
        seed = 42
        ntrain = 6
        ndev = 3
        did2doc = {doc.doc_id: doc for doc in msmarco.loaded_data.corpus}
        random_state = random.Random(seed)

        os.makedirs(output_dir, exist_ok=True)
        sampled_doc_ids = set()
        lqs: List[LabeledQuery]

        # Build and save example files for labeled queries:
        for fdata, fto, lqs, n in [
            (
                msmarco.fdev,
                os.path.join(output_dir, "dev_v2.1.json"),
                msmarco.loaded_data.labeled_queries_test,
                ndev,
            ),
            (
                msmarco.ftrain,
                os.path.join(output_dir, "train_v2.1.json"),
                msmarco.loaded_data.labeled_queries_train,
                ntrain,
            ),
        ]:
            df = pd.read_json(fdata)
            lqs_sampled = random_state.sample(lqs, n)
            query_ids = [lq.query.query_id for lq in lqs_sampled]
            for lq in lqs_sampled:
                for jchk in lq.judged_chunks:
                    sampled_doc_ids.add(jchk.chunk.belonging_doc.doc_id)

            subset = df[df["query_id"].isin(query_ids)]
            subset.to_json(fto)

        # Build and save corresponding corpus file:
        drecords = msmarco._build_document_records()
        drecords_subset: List[DocumentRecord] = []
        for drecord in drecords:
            if drecord.doc_id in sampled_doc_ids:
                drecords_subset.append(drecord)
        fcorpus = os.path.join(output_dir, "msmarco-docs.tsv")
        with open(fcorpus, "w") as f:
            for drecord in drecords_subset:
                line = (
                    "\t".join(
                        [drecord.doc_id, drecord.url, drecord.title, drecord.text]
                    )
                    + "\n"
                )
                f.write(line)
