from dataclasses import dataclass
from functools import partial
from typing import Dict, Iterable, List, Optional
from clddp.dm import RetrievalDataset, Passage, JudgedPassage, Query, LabeledQuery
from dapr.datasets.dm import LoadedData
from dapr.datasets.dm import LabeledQuery as DAPRLabeledQuery
from dapr.dm import RetrievalLevel, ParagraphSeparator
import tqdm


@dataclass
class DAPRDataConfig:
    data_name_or_path: str
    titled: bool = False
    retrieval_level: RetrievalLevel = RetrievalLevel.paragraph
    paragraph_separator: ParagraphSeparator = ParagraphSeparator.blank


class DAPRDataLoader:
    def __init__(self, config: DAPRDataConfig) -> None:
        self.config = config

    def collection_iter_fn(
        self,
        data: LoadedData,
    ) -> Iterable[Passage]:
        titled = self.config.titled
        retrieval_level = self.config.retrieval_level
        paragraph_separator = self.config.paragraph_separator
        for doc in data.corpus_iter_fn():
            doc.set_default_candidates()
            title = doc.title if titled else None
            if retrieval_level is RetrievalLevel.document:
                yield Passage(
                    passage_id=doc.doc_id,
                    text=paragraph_separator.string.join(
                        chk.text for chk in doc.chunks
                    ),
                    title=title,
                )
            else:
                for chunk in doc.chunks:
                    if chunk.chunk_id in doc.candidate_chunk_ids:
                        yield Passage(
                            passage_id=chunk.chunk_id,
                            text=chunk.text,
                            title=chunk.belonging_doc.title if titled else None,
                        )

    def build_labeled_queries(
        self,
        labeled_queries: Optional[List[DAPRLabeledQuery]],
    ) -> List[LabeledQuery]:
        titled = self.config.titled
        retrieval_level = self.config.retrieval_level
        paragraph_separator = self.config.paragraph_separator
        if labeled_queries is None:
            return None
        lqs = []
        for lq in labeled_queries:
            pid2positive: Dict[str, JudgedPassage] = {}
            pid2negative: Dict[str, JudgedPassage] = {}
            for jchk in lq.judged_chunks:
                query = Query(query_id=jchk.query.query_id, text=jchk.query.text)
                chunk = jchk.chunk
                doc = chunk.belonging_doc
                title = doc.title if titled else None
                if retrieval_level is RetrievalLevel.document:
                    pid = doc.doc_id
                    passage = Passage(
                        passage_id=pid,
                        text=paragraph_separator.string.join(
                            chk.text for chk in doc.chunks
                        ),
                        title=title,
                    )
                else:
                    pid = chunk.chunk_id
                    passage = Passage(
                        passage_id=pid,
                        text=chunk.text,
                        title=chunk.belonging_doc.title if titled else None,
                    )
                jpsg = JudgedPassage(
                    query=query, passage=passage, judgement=jchk.judgement
                )
                if jchk.judgement:
                    # For document-level annotation, keep the highest judgement on paragraphs:
                    if (
                        pid not in pid2positive
                        or pid2positive[pid].judgement < jpsg.judgement
                    ):
                        pid2positive[pid] = jpsg
                else:
                    pid2negative[pid] = jpsg
                lqs.append(
                    LabeledQuery(
                        query=query,
                        positives=list(pid2positive.values()),
                        negatives=list(pid2negative.values()),
                    )
                )
        return lqs

    def load_data(self, progress_bar: bool) -> RetrievalDataset:
        data = LoadedData.from_dump(self.config.data_name_or_path, pbar=progress_bar)
        assert data.meta_data is not None
        retrieval_level = self.config.retrieval_level
        if retrieval_level is RetrievalLevel.document:
            collection_size = data.meta_data["ndocs"]
        else:
            collection_size = (
                data.meta_data["nchunks"]
                if data.meta_data.get("nchunks_candidates") is None
                else data.meta_data["nchunks_candidates"]
            )
        dataset = RetrievalDataset(
            collection_iter_fn=partial(self.collection_iter_fn, data=data),
            collection_size=collection_size,
            train_labeled_queries=self.build_labeled_queries(
                labeled_queries=data.labeled_queries_train
            ),
            dev_labeled_queries=self.build_labeled_queries(
                labeled_queries=data.labeled_queries_dev
            ),
            test_labeled_queries=self.build_labeled_queries(
                labeled_queries=data.labeled_queries_test
            ),
        )
        return dataset

    def get_pid2did(self, progress_bar: bool) -> Dict[str, str]:
        data = LoadedData.from_dump(self.config.data_name_or_path, pbar=progress_bar)
        pid2did = {}
        assert data.corpus_iter_fn is not None
        for doc in tqdm.tqdm(
            data.corpus_iter_fn(),
            total=data.meta_data["ndocs"],
            desc="Building pid2did",
            disable=not progress_bar,
        ):
            for chunk in doc.chunks:
                pid2did[chunk.chunk_id] = doc.doc_id
        return pid2did

    def get_did2pids(self, progress_bar: bool) -> Dict[str, List[str]]:
        data = LoadedData.from_dump(self.config.data_name_or_path, pbar=progress_bar)
        did2pids: Dict[str, List[str]] = {}
        assert data.corpus_iter_fn is not None
        for doc in tqdm.tqdm(
            data.corpus_iter_fn(),
            total=data.meta_data["ndocs"],
            desc="Building did2pids",
            disable=not progress_bar,
        ):
            for chunk in doc.chunks:
                did2pids.setdefault(doc.doc_id, [])
                did2pids[doc.doc_id].append(chunk.chunk_id)
        return did2pids
