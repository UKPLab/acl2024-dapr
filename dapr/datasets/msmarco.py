from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, replace
import os
import random
from typing import Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch

import datasets
from dapr.datasets.base import BaseDataset, LoadedData
from dapr.datasets.dm import Chunk, Document, JudgedChunk, LabeledQuery, Query, Split
from dapr.utils import (
    download,
    tqdm_ropen,
    Multiprocesser,
    set_logger_format,
    cache_to_disk,
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

    @staticmethod
    def build_title2urls(drecords: List[DocumentRecord]) -> Dict[str, List[str]]:
        title2urls: Dict[str, List[str]] = {}
        for drecord in drecords:
            title2urls.setdefault(drecord.title, list())
            title2urls[drecord.title].append(drecord.url)
        return title2urls

    @staticmethod
    def build_url2drecod(drecords: List[DocumentRecord]) -> Dict[str, str]:
        return {drecord.url: drecord for drecord in drecords}


@dataclass
class PassageRecord:
    query_id: str
    query: str
    # is_selected: bool  # Keep only positive
    url: str
    passage_text: str
    span: Optional[Tuple[int, int]] = None
    judgement: Optional[int] = None

    @staticmethod
    def build_psg2url(precords: List[PassageRecord]) -> Dict[str, str]:
        return {precord.passage_text: precord.url for precord in precords}


@dataclass
class TitledPassage:
    title: str
    passage: str
    passage_id: str

    @staticmethod
    def build_pid2tpsg(
        titled_passages: List[TitledPassage],
    ) -> Dict[str, TitledPassage]:
        return {tpsg.passage_id: tpsg for tpsg in titled_passages}


class MSMARCO(BaseDataset):
    """MSMARCO: Load the dataset into corpus and labeled queries."""

    fcorpus: Optional[str] = None
    ftrain_qna: Optional[str] = None  # Will become the train split
    fdev_qna: Optional[str] = (
        None  # Just for linking the TRECDL passages to the documents
    )
    feval_qna: Optional[str] = (
        None  # Just for linking the TRECDL passages to the documents
    )
    ftitles_pranking: Optional[str] = None
    fpassages_pranking: Optional[str] = None
    fqrels_small: Optional[str] = None  # Will become the test split
    ftrecdl19_queries: Optional[str] = None  # Will become the dev split
    ftrecdl19_qrels: Optional[str] = None
    ftrecdl20_queries: Optional[str] = None  # Will become the test split
    ftrecdl20_qrels: Optional[str] = None
    max_mismatch: int = 5
    EMPTY_TITLES: Set[str] = {"", ".", "-"}

    def _download(self, resource_path: str) -> None:
        """The resource path should be something like https://msmarco.blob.core.windows.net."""
        if os.path.exists(resource_path):
            self.fcorpus = os.path.join(resource_path, "msmarco-docs.tsv")
            self.ftrain_qna = os.path.join(resource_path, "train_v2.1.json")
            self.fdev_qna = os.path.join(resource_path, "dev_v2.1.json")
            self.feval_qna = os.path.join(resource_path, "eval_v2.1_public.json")
            self.ftitles_pranking = os.path.join(resource_path, "para.title.txt")
            self.fpassages_pranking = os.path.join(resource_path, "para.txt")
            self.fqrels_small = os.path.join(resource_path, "qrels.dev.small.tsv")
            self.ftrecdl19_queries = os.path.join(
                resource_path, "msmarco-test2019-queries.tsv"
            )
            self.ftrecdl19_qrels = os.path.join(resource_path, "2019qrels-pass.txt")
            self.ftrecdl20_queries = os.path.join(
                resource_path, "msmarco-test2020-queries.tsv"
            )
            self.ftrecdl20_qrels = os.path.join(resource_path, "2020qrels-pass.txt")
        else:
            resource_path = "https://msmarco.blob.core.windows.net"
            fcorpus = os.path.join(resource_path, "msmarcoranking/msmarco-docs.tsv.gz")
            ftrain_qna = os.path.join(resource_path, "msmarco/train_v2.1.json.gz")
            fdev_qna = os.path.join(resource_path, "msmarco/dev_v2.1.json.gz")
            feval_qna = os.path.join(resource_path, "msmarco/eval_v2.1_public.json.gz")
            fqres_small = os.path.join(resource_path, "qrels.dev.small.tsv")
            rocketqa_msmarco = "https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz"
            ftrecdl19_queries = os.path.join(
                resource_path, "msmarcoranking/msmarco-test2019-queries.tsv.gz"
            )
            ftrecdl19_qrels = "https://trec.nist.gov/data/deep/2019qrels-pass.txt"
            ftrecdl20_queries = os.path.join(
                resource_path, "msmarcoranking/msmarco-test2020-queries.tsv.gz"
            )
            ftrecdl20_qrels = "https://trec.nist.gov/data/deep/2020qrels-pass.txt"
            with patch.object(datasets.utils.file_utils, "http_get", download):
                dm = DownloadManager()
                self.fcorpus = dm.download_and_extract(fcorpus)
                self.ftrain_qna = dm.download_and_extract(ftrain_qna)
                self.fdev_qna = dm.download_and_extract(fdev_qna)
                self.feval_qna = dm.download_and_extract(feval_qna)
                self.fqres_small = dm.download_and_extract(fqres_small)
                rocketqa_msmarco_dir = dm.download_and_extract(rocketqa_msmarco)
                self.ftitles_pranking = os.path.join(
                    rocketqa_msmarco_dir, "marco", "para.title.txt"
                )
                self.fpassages_pranking = os.path.join(
                    rocketqa_msmarco_dir, "marco", "para.txt"
                )
                self.ftrecdl19_queries = dm.download_and_extract(ftrecdl19_queries)
                self.ftrecdl19_qrels = dm.download(ftrecdl19_qrels)
                self.ftrecdl20_queries = dm.download_and_extract(ftrecdl20_queries)
                self.ftrecdl20_qrels = dm.download(ftrecdl20_qrels)

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

    def _read_qna_records(self, fpath: str, relevant_only: bool) -> List[PassageRecord]:
        self.logger.info(f"Reading QNA records from {os.path.basename(fpath)}")
        df = pd.read_json(fpath)
        records: List[PassageRecord] = []
        for _, line_dict in df.iterrows():
            for psg in line_dict["passages"]:
                record = PassageRecord(
                    query_id=str(line_dict["query_id"]),
                    query=line_dict["query"],
                    url=psg["url"],
                    passage_text=psg["passage_text"],
                    judgement=psg.get("is_selected", None),
                )
                records.append(record)
        if relevant_only:
            records = list(filter(lambda r: r.judgement, records))
        return records

    def _read_pranking_dev_qids(self) -> Set[str]:
        qids = set()
        for line in tqdm_ropen(self.fqrels_small, desc="Reading Passage Ranking qrels"):
            qid, _, pid, judgment = line.strip().split()
            if str(judgment) == 0:
                continue
            qids.add(qid)
        return qids

    def _read_pranking_data(self) -> List[TitledPassage]:
        pids = []
        titles = []
        for line in tqdm_ropen(
            fpath=self.ftitles_pranking, desc=f"Reading Passage Ranking titles"
        ):
            pid, title = line.strip().split("\t")
            pids.append(pid)
            titles.append(title)

        passages = []
        line: str
        for i, line in enumerate(
            tqdm_ropen(
                fpath=self.fpassages_pranking,
                desc=f"Reading Passage Ranking passages {os.path.basename(self.fcorpus)}",
            )
        ):
            pid, passage = line.strip().split("\t")
            passages.append(passage)
            assert (
                pids[i] == pid
            ), f"{self.ftitles_pranking} and {self.fpassages_pranking} are not aligned."

        titled_passages = []
        for pid, title, passage in zip(pids, titles, passages):
            titled_passage = TitledPassage(title=title, passage=passage, passage_id=pid)
            titled_passages.append(titled_passage)
        return titled_passages

    def _read_trecdl_records(
        self,
        fqueries: str,
        fqrels: str,
        title2urls: Dict[str, List[str]],
        titled_passages: List[TitledPassage],
        psg2url: Dict[str, str],
    ) -> List[PassageRecord]:
        qid2query: Dict[str, str] = {}
        with open(fqueries) as f:
            for line in f:
                qid, query = line.split("\t")
                query = query.strip()
                qid2query[qid] = query

        precords = []
        pid2titled_passage: Dict[str, TitledPassage] = {
            tp.passage_id: tp for tp in titled_passages
        }
        with open(fqrels) as f:
            for line in f:
                qid, _, pid, judgement = line.strip().split()
                judgement = int(judgement)
                if judgement == 0:
                    continue

                query = qid2query[qid]
                tpsg = pid2titled_passage[pid]
                url = psg2url.get(tpsg.passage, None)
                if url:
                    psg2url[tpsg.passage]
                    precord = PassageRecord(
                        query_id=qid,
                        query=query,
                        url=url,
                        passage_text=tpsg.passage,
                        judgement=judgement,
                    )
                    precords.append(precord)
                    continue

                if tpsg.title in self.EMPTY_TITLES:
                    # Otherwise it will be too costly
                    continue

                if tpsg.title not in title2urls:
                    continue

                urls = title2urls[tpsg.title]
                for url in urls:
                    precord = PassageRecord(
                        query_id=qid,
                        query=query,
                        url=url,
                        passage_text=tpsg.passage,
                        judgement=judgement,
                    )
                    precords.append(precord)
        return precords

    def _filter_document_records(
        self,
        document_records: List[DocumentRecord],
        titles: Set[str],
        labeled_queries: List[LabeledQuery],
    ) -> List[DocumentRecord]:
        """The final document should either have titles from the passage ranking dataset or included in the labeled queries."""
        doc_ids_in_lqs = {
            jchk.chunk.belonging_doc.doc_id
            for lq in labeled_queries
            for jchk in lq.judged_chunks
        }
        records = list(
            filter(
                lambda doc: doc.title in titles or doc.doc_id in doc_ids_in_lqs,
                tqdm.tqdm(document_records, desc="Filtering document records"),
            )
        )
        return records

    def _remove_documents_wrt_nonmatched(
        self, document_records: List[DocumentRecord], precords: List[PassageRecord]
    ) -> List[DocumentRecord]:
        """Remove the documents corresponding to nonmatched passage records. We do this for reducing nonannotated queries."""
        url2matched: Dict[str, bool] = {}
        for precord in precords:
            url2matched.setdefault(precord.url, False)
            if precord.span is not None:
                url2matched[precord.url] = True
        url_to_exclude = {url for url, matched in url2matched.items() if not matched}
        self.logger.info(
            f"Going to exclude {len(url_to_exclude)} URLs due to no match in the passage records."
        )
        drecords = list(
            filter(lambda drecord: drecord.url not in url_to_exclude, document_records)
        )
        return drecords

    def _filter_passage_records(
        self, passage_records: List[PassageRecord], urls: Set[str]
    ) -> List[PassageRecord]:
        """Filter the QNA records to make sure all the passage urls are in the given url set."""
        records = list(
            filter(
                lambda psg: psg.url in urls,
                tqdm.tqdm(passage_records, desc="Filtering passage records"),
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
            chunk_size=50000,
        )
        return results

    def _build_document_and_judged_chunks(
        self,
        drecord: DocumentRecord,
        precords: List[PassageRecord],
    ) -> Tuple[Document, List[JudgedChunk]]:
        seed = int(drecord.doc_id.strip("D"))  # D123 -> 123
        query: Optional[Query] = None

        # Build the spans:
        precords_sorted: List[PassageRecord] = sorted(
            precords, key=lambda precord: precord.span
        )
        queries: List[Optional[Query]] = []
        texts: List[str] = []
        judgements: List[Optional[int]] = []
        b = 0
        for precord in precords_sorted:
            e = precord.span[0]
            assert e >= b, "Overlapping cases not removed"
            if e > b:
                queries.append(None)
                texts.append(drecord.text[b:e])
                judgements.append(None)
            b = e
            e = precord.span[1]
            assert e >= b, "Overlapping cases not removed"
            if e > b:
                queries.append(Query(query_id=precord.query_id, text=precord.query))
                texts.append(drecord.text[b:e])
                judgements.append(precord.judgement)
            b = e
        assert b <= len(drecord.text), "Span end out of document"
        if b < len(drecord.text):
            e = len(drecord.text)
            queries.append(None)
            texts.append(drecord.text[b:e])
            judgements.append(None)

        # Build the chunks:
        document = Document(
            doc_id=drecord.doc_id,
            chunks=[],
            title=drecord.title,
            candidate_chunk_ids=set(),
        )
        judged_chunks = []
        for query, text, judgement in zip(queries, texts, judgements):
            chunk_id = Chunk.build_chunk_id(
                doc_id=drecord.doc_id, position=len(document.chunks)
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                text=text,
                doc_summary=None,
                belonging_doc=document,
            )
            document.chunks.append(chunk)
            if judgement is not None:
                document.candidate_chunk_ids.add(chunk.chunk_id)
            if judgement and query is not None:
                judged_chunks.append(
                    JudgedChunk(query=query, chunk=chunk, judgement=precord.judgement)
                )

        return document, judged_chunks

    def _build_url2precords(
        self, passage_records: List[PassageRecord]
    ) -> Dict[str, List[PassageRecord]]:
        url2precords: Dict[str, List[PassageRecord]] = {}
        for precord in passage_records:
            if precord.span is None:
                continue
            url2precords.setdefault(precord.url, [])
            url2precords[precord.url].append(precord)
        return url2precords

    def _build_corpus(
        self, drecords: List[DocumentRecord], precords: List[PassageRecord]
    ) -> Union[List[Document], Dict[str, List[JudgedChunk]]]:
        url2precords = self._build_url2precords(precords)
        corpus = []
        qid2jchks: Dict[str, List[JudgedChunk]] = (
            {}
        )  # Sometimes multiple records are labeled on the same document
        for drecord in drecords:
            if drecord.url not in url2precords:
                continue
            doc, jchks = self._build_document_and_judged_chunks(
                drecord=drecord, precords=url2precords[drecord.url]
            )
            for jchk in jchks:
                qid2jchks.setdefault(jchk.query.query_id, [])
                qid2jchks[jchk.query.query_id].append(jchk)
            corpus.append(doc)
        return corpus, qid2jchks

    def _build_labeled_queries(
        self,
        passage_records: List[PassageRecord],
        url2drecord: Dict[str, DocumentRecord],
    ) -> List[LabeledQuery]:
        url2precords = self._build_url2precords(passage_records)
        qid2jchunks: Dict[str, List[JudgedChunk]] = defaultdict(list)
        for drecord in url2drecord.values():
            if drecord.url not in url2precords:
                continue
            _, jchks = self._build_document_and_judged_chunks(
                drecord=drecord, precords=url2precords[drecord.url]
            )
            for jchk in jchks:
                qid2jchunks[jchk.query.query_id].append(jchk)
        labeled_queries = [
            LabeledQuery(query=jchunks[0].query, judged_chunks=jchunks)
            for jchunks in qid2jchunks.values()
        ]
        return labeled_queries

    def _remove_nonmatched(self, precords: List[PassageRecord]) -> List[PassageRecord]:
        return [precord for precord in precords if precord.span is not None]

    def _remove_duplicate_precords(
        self, precords: List[PassageRecord]
    ) -> List[PassageRecord]:
        existed: Dict[str, Set[Tuple[int, int]]] = {}
        processed: List[PassageRecord] = []
        for precord in tqdm.tqdm(precords, desc="Removing duplicate passage records"):
            if precord.url in existed:
                if precord.span in existed[precord.url]:
                    continue
            processed.append(precord)
            existed.setdefault(precord.url, set())
            existed[precord.url].add(precord.span)
        return processed

    def _remove_overlapping_precords(
        self,
        precords_train: List[PassageRecord],
        precords_dev: List[PassageRecord],
        precords_test: List[PassageRecord],
    ) -> None:
        """Remove passage records which have overlapping spans with one another within the same doc."""
        self.logger.info("Removing overlapping passage records")
        url2split_precords: Dict[str, List[Tuple[Split, PassageRecord]]] = {}
        for precords, split in zip(
            [precords_train, precords_dev, precords_test],
            [Split.train, Split.dev, Split.test],
        ):
            for precord in precords:
                if precord.span:
                    url2split_precords.setdefault(precord.url, [])
                    url2split_precords[precord.url].append((split, precord))

        to_pop: Dict[str, Set[Tuple[Split, int, int]]] = {}
        for split_precords in tqdm.tqdm(
            url2split_precords.values(),
            total=len(url2split_precords),
            desc="Finding overlappings",
        ):
            if len(split_precords) < 2:
                continue

            sorted_split_precords: List[Tuple[Split, PassageRecord]] = sorted(
                split_precords, key=lambda x: x[1].span
            )
            cur = sorted_split_precords.pop(0)
            while len(sorted_split_precords):
                next = sorted_split_precords.pop(0)
                if cur[1].span[1] > next[1].span[0]:
                    # Order: test -> dev -> train
                    split_precord_to_pop = None
                    if cur[0] is Split.test:
                        split_precord_to_pop = next
                    elif next[0] is Split.test:
                        split_precord_to_pop = cur
                        cur = next
                    elif cur[0] is Split.dev:
                        split_precord_to_pop = next
                    elif next[0] is Split.dev:
                        split_precord_to_pop = cur
                        cur = next
                    else:
                        assert cur[0] is Split.train
                        assert next[0] is Split.train
                        split_precord_to_pop = cur
                        cur = next
                    to_pop.setdefault(split_precord_to_pop[1].url, set())
                    to_pop[split_precord_to_pop[1].url].add(
                        (split_precord_to_pop[0],) + split_precord_to_pop[1].span
                    )
                else:
                    cur = next

        for precords, split in zip(
            [precords_train, precords_dev, precords_test],
            [Split.train, Split.dev, Split.test],
        ):
            for i in tqdm.trange(
                len(precords) - 1, -1, -1, desc="Removing overlappings"
            ):
                precord = precords[i]
                if precord.url in to_pop:
                    if (split,) + precord.span in to_pop[precord.url]:
                        precords.pop(i)

    def _replace_judged_chunks(
        self,
        labeled_queries: List[LabeledQuery],
        qid2jchks: Dict[str, List[JudgedChunk]],
    ) -> None:
        """Replace the judged chunks to make it align with the corpus."""
        for lq in labeled_queries:
            if lq.query.query_id not in qid2jchks:
                continue

            jchks = qid2jchks[lq.query.query_id]
            lq.judged_chunks = jchks

    def _remove_negatives(
        self, labeled_queries: List[LabeledQuery]
    ) -> List[LabeledQuery]:
        processed = []
        for lq in labeled_queries:
            jchks = [jchk for jchk in lq.judged_chunks if jchk.judgement]
            if len(jchks):
                processed.append(LabeledQuery(query=lq.query, judged_chunks=jchks))
        return processed

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        # Load and filter data:
        drecords = self._build_document_records()
        train_records_qna = self._read_qna_records(
            fpath=self.ftrain_qna, relevant_only=False
        )
        dev_records_qna = self._read_qna_records(
            fpath=self.fdev_qna, relevant_only=False
        )
        url2drecord = DocumentRecord.build_url2drecod(drecords)
        urls = set(url2drecord)
        pranking_dev_qids = self._read_pranking_dev_qids()
        heldout_records = cache_to_disk(cache_path="msmarco_heldout_records", tmp=True)(
            self._find_matches
        )(
            precords=self._filter_passage_records(
                passage_records=dev_records_qna, urls=urls
            ),
            drecords=drecords,
        )
        heldout_records = self._remove_duplicate_precords(
            self._remove_nonmatched(heldout_records)
        )
        test_records = [
            record
            for record in heldout_records
            if record.query_id in pranking_dev_qids and record.judgement
        ]
        random_state = random.Random(42)
        dev_candidate_records = [
            record
            for record in heldout_records
            if record.query_id not in pranking_dev_qids and record.judgement
        ]
        dev_records = random_state.sample(
            dev_candidate_records, k=min(len(test_records), len(dev_candidate_records))
        )  # There are many more queries besides the passage-ranking dev queries, which can be used as the new dev queries
        train_records = cache_to_disk(cache_path="msmarco_train_records", tmp=True)(
            self._find_matches
        )(
            precords=self._filter_passage_records(
                passage_records=train_records_qna, urls=urls
            ),
            drecords=drecords,
        )
        dev_records = self._remove_duplicate_precords(
            self._remove_nonmatched(dev_records)
        )
        test_records = self._remove_duplicate_precords(
            self._remove_nonmatched(test_records)
        )
        train_records = self._remove_duplicate_precords(
            self._remove_nonmatched(train_records)
        )
        self._remove_overlapping_precords(
            precords_train=train_records,
            precords_dev=dev_records,
            precords_test=test_records,
        )

        # Build corpus and labeled queries:
        labeled_queries_test = self._build_labeled_queries(
            passage_records=test_records, url2drecord=url2drecord
        )
        labeled_queries_dev = self._build_labeled_queries(
            passage_records=dev_records, url2drecord=url2drecord
        )
        labeled_queries_train = self._build_labeled_queries(
            passage_records=train_records, url2drecord=url2drecord
        )
        drecords = self._remove_documents_wrt_nonmatched(
            document_records=drecords,
            precords=train_records + dev_records + test_records,
        )
        corpus, qid2jchks = self._build_corpus(
            drecords=drecords, precords=train_records + dev_records + test_records
        )
        self._replace_judged_chunks(
            labeled_queries=labeled_queries_train, qid2jchks=qid2jchks
        )
        self._replace_judged_chunks(
            labeled_queries=labeled_queries_dev, qid2jchks=qid2jchks
        )
        self._replace_judged_chunks(
            labeled_queries=labeled_queries_test, qid2jchks=qid2jchks
        )
        labeled_queries_train = self._remove_negatives(labeled_queries_train)
        labeled_queries_dev = self._remove_negatives(labeled_queries_dev)
        labeled_queries_test = self._remove_negatives(labeled_queries_test)
        return LoadedData(
            corpus_iter_fn=lambda: iter(corpus),
            labeled_queries_train=labeled_queries_train,
            labeled_queries_dev=labeled_queries_dev,
            labeled_queries_test=labeled_queries_test,
        )


if __name__ == "__main__":
    set_logger_format()
    msmarco = MSMARCO(
        # resource_path="https://msmarco.blob.core.windows.net",
        resource_path="/home/fb20user07/research/msmarco/all",
        nheldout=None,
    )
