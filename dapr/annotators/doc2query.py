from dataclasses import dataclass
import logging
import os
import sys
from typing import Dict, Iterable, List, Optional, Set, TypedDict, Union
from dapr.annotators.base import BaseAnnotator
from dapr.datasets.base import BaseDataset, LoadedData
import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.context import SpawnProcess
from dapr.datasets.dm import Document
from dapr.utils import tqdm_ropen
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    AutoTokenizer,
    ElectraForSequenceClassification,
)
import more_itertools

import numpy as np
import tqdm
import ujson
from functools import cached_property


class T5TokenizerFastCached(T5TokenizerFast):
    @cached_property
    def all_special_ids(self) -> Set[int]:
        return set(super().all_special_ids)


class QueryGenerator:
    def __init__(self, device: Union[int, str], nsamples: int) -> None:
        model_name_or_path = "macavaney/doc2query-t5-base-msmarco"
        self.tokenizer = T5TokenizerFastCached.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.nsamples = nsamples

    @torch.no_grad()
    def run(self, text_batch: List[str]) -> List[List[str]]:
        tokenized = self.tokenizer(
            text_batch,
            max_length=64,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids: torch.Tensor = tokenized["input_ids"]
        outputs = self.model.generate(
            input_ids=input_ids.to(self.device),
            max_length=64,
            do_sample=True,
            top_k=10,
            num_return_sequences=self.nsamples,
        )
        gqs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        query_lists = list(more_itertools.chunked(gqs, n=self.nsamples))
        return query_lists


class QueryFilter:
    def __init__(
        self, device: Union[int, str], keep_ratio: float, batch_size: int
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/electra-base-discriminator"
        )
        self.model = ElectraForSequenceClassification.from_pretrained(
            "crystina-z/monoELECTRA_LCE_nneg31"
        )
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.keep_ratio = keep_ratio
        self.batch_size = batch_size

    @torch.no_grad()
    def run(self, passage: str, queries: List[str]) -> List[str]:
        scores = []
        nkept = int(len(queries) * self.keep_ratio)
        assert len(queries) > nkept
        for b in range(0, len(queries), self.batch_size):
            e = b + self.batch_size
            query_batch = queries[b:e]
            tokenized = self.tokenizer(
                query_batch,
                [passage] * len(query_batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            scores.append(self.model(**tokenized)["logits"][:, 1])
        scores = torch.cat(scores)
        indices_kept: torch.LongTensor = scores.topk(k=nkept)[1]
        filtered: np.ndarray = np.array(queries)[indices_kept.cpu()]
        return filtered.tolist()


class PyTerrierRow(TypedDict):
    docno: str
    text: str


@dataclass
class WorkerOutput:
    cid2gqs: Dict[str, str]
    nentries: int


class ChunkID2GeneratedQueriesJson(TypedDict):
    chunk_id: str
    generated_queries: str


class Doc2QueryAnnotator(BaseAnnotator):
    """Doc2Query-- method. The generated queries will be appended after each chunk text. The doc_summary will be left empty."""

    def __init__(self, nsamples: int, keep_ratio: float, batch_size_chunk: int) -> None:
        self.nsamples = nsamples
        self.keep_ratio = keep_ratio
        self.batch_size_chunk = batch_size_chunk
        self.logger = logging.getLogger(__name__)
        self.ps: Optional[List[SpawnProcess]] = None

    # TODO: Merge these into base.py
    def annotate(self, data: LoadedData, cache_root_dir: str) -> None:
        """Annotate the doc_summary fields inplace."""
        data.meta_data["corpus_identifier"] = "/".join(
            [
                data.meta_data["corpus_identifier"],
                self.__class__.__name__,
            ]
        )
        cache_fpath = os.path.join(
            cache_root_dir,
            data.meta_data["corpus_identifier"],
            "cid2gqs.jsonl",
        )
        if os.path.exists(cache_fpath):
            cid2gqs: Dict[str, str] = {}
            for line in tqdm_ropen(
                fpath=cache_fpath, desc="Loading document summaries"
            ):
                line_dict: ChunkID2GeneratedQueriesJson = ujson.loads(line)
                cid2gqs[line_dict["chunk_id"]] = line_dict["generated_queries"]
        else:
            os.makedirs(os.path.dirname(cache_fpath), exist_ok=True)
            try:
                cid2gqs = self._annotate(dataset)
                with open(cache_fpath, "w") as f:
                    for cid, gqs in cid2gqs.items():
                        line_dict = ChunkID2GeneratedQueriesJson(
                            chunk_id=cid, generated_queries=gqs
                        )
                        line = ujson.dumps(line_dict) + "\n"
                        f.write(line)
            except Exception as e:
                if os.path.exists(cache_fpath):
                    os.remove(cache_fpath)
                raise e

        for lqs in [
            data.labeled_queries_train,
            data.labeled_queries_dev,
            data.labeled_queries_test,
        ]:
            if lqs is None:
                continue
            for lq in lqs:
                for jchk in lq.judged_chunks:
                    jchk.chunk.doc_summary = cid2gqs[jchk.chunk.chunk_id]

        corpus_iter_fn = data.corpus_iter_fn

        def new_corpus_iter_fn() -> Iterable[Document]:
            for doc in corpus_iter_fn():
                for chk in doc.chunks:
                    chk.text = "\n".join(cid2gqs[chk.chunk_id]) + "\n" + chk.text
                yield doc

        data.corpus_iter_fn = new_corpus_iter_fn

    def _annotate(self, data: LoadedData) -> Dict[str, str]:
        """Annotate the dataset and return cid2gqs."""
        cid2gqs = self.process(
            pool=data.corpus_iter_fn(),
            ndocs=data.meta_data["ndocs"],
            nchunks=data.meta_data["nchunks"],
        )
        doc_ids = {doc.doc_id for doc in data.corpus_iter_fn()}

        leftover: List[Document] = []
        for lqs in [
            data.labeled_queries_dev,
            data.labeled_queries_test,
            data.labeled_queries_train,
        ]:
            if lqs is None:
                continue
            for lq in lqs:
                for jchk in lq.judged_chunks:
                    doc = jchk.chunk.belonging_doc
                    if doc.doc_id not in doc_ids:
                        leftover.append(doc)
        nchunks = sum(len(doc.chunks) for doc in leftover)
        cid2gqs_leftover = self.process(
            pool=leftover, ndocs=len(leftover), nchunks=nchunks
        )
        cid2gqs.update(cid2gqs_leftover)
        return cid2gqs

    def process(
        self, pool: Iterable[Document], ndocs: int, nchunks: int
    ) -> Dict[str, str]:
        """Process the pool and return cid2gqs."""
        dbatches = Document.split_pool(
            pool=pool,
            batch_size_chunk=self.batch_size_chunk,
        )

        ctx = mp.get_context("spawn")
        qin = ctx.Queue()
        qout = ctx.Queue()
        ngpus = torch.cuda.device_count()
        self.ps = [
            ctx.Process(
                target=self.worker,
                args=(gpu, qin, qout, self.nsamples, self.keep_ratio),
                daemon=True,
            )
            for gpu in range(ngpus)
        ]
        pbar = tqdm.tqdm(total=nchunks, desc="Doing Doc2Query--")
        # reduce_per = 5000
        reduce_per = 50000
        for p in tqdm.tqdm(self.ps, desc="Starting processes"):
            p.start()

        cid2gqs: Dict[str, str] = {}
        chunk_ids: List[str] = []
        did_per_chunk: List[str] = []
        doc_ids: List[str] = []
        nput: int = 0
        try:
            for i, dbatch in enumerate(dbatches):
                qin.put(dbatch)
                nput += 1
                for doc in dbatch:
                    doc_ids.append(doc.doc_id)
                    for chunk in doc.chunks:
                        chunk_ids.append(chunk.chunk_id)
                        did_per_chunk.append(doc.doc_id)

                if (i + 1) % reduce_per == 0 or len(doc_ids) == ndocs:
                    for _ in range(nput):
                        self.watch_alive()
                        output: WorkerOutput = qout.get()
                        cid2gqs.update(output.cid2gqs)
                        pbar.update(output.nentries)
                    nput = 0  # Remember to empty nput after processing!
        finally:
            for p in self.ps:
                p.kill()

        return cid2gqs

    def watch_alive(self) -> None:
        for p in self.ps:
            if not p.is_alive():
                for p in self.ps:
                    p.terminate()
                self.logger.info("Exit due to error in subprocess")
                sys.exit()

    @staticmethod
    def worker(
        device: int, qin: Queue, qout: Queue, nsamples: int, keep_ratio: float
    ) -> None:
        query_generator = QueryGenerator(device=device, nsamples=nsamples)
        query_filter = QueryFilter(
            device=device, keep_ratio=keep_ratio, batch_size=nsamples
        )
        batch_size = 32
        logger = logging.getLogger(__name__)
        while True:
            try:
                dbatch: List[Document] = qin.get()
                chunks = [chunk for doc in dbatch for chunk in doc.chunks]
                texts = list(map(lambda chunk: chunk.text, chunks))
                cid2gqs = {}
                for b in range(0, len(texts), batch_size):
                    with torch.cuda.amp.autocast():
                        e = b + batch_size
                        qlists = query_generator.run(texts[b:e])
                        assert len(qlists) == len(chunks[b:e])
                        for chunk, qlist in zip(chunks[b:e], qlists):
                            filtered = query_filter.run(
                                passage=chunk.text, queries=qlist
                            )
                            cid2gqs[chunk.chunk_id] = filtered
                qout.put(WorkerOutput(cid2gqs=cid2gqs, nentries=len(chunks)))
            except Exception as e:
                logger.error(str(e))
                raise e
