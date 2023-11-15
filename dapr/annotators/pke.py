from __future__ import annotations
from enum import Enum
from functools import partial
from typing import Dict, Iterable, List, Tuple
from dapr.datasets.base import LoadedData
from dapr.datasets.dm import Document
from dapr.utils import Multiprocesser
from pke.base import LoadFile
from pke.unsupervised import TopicRank
from dapr.annotators.base import BaseAnnotator

MAX_LENGTH = 65536  # <= 10*6 Required by spaCy, 65536 takes around 12s already


class KeyphraseApproach(str, Enum):
    topic_rank = "topic_rank"

    def __call__(self, text: str, n: int) -> Dict[str, float]:
        model = {KeyphraseApproach.topic_rank: TopicRank}[self]()
        keyphrases = {KeyphraseApproach.topic_rank: KeyphraseApproach._topic_rank}[
            self
        ](model, text, n)
        return keyphrases

    @staticmethod
    def _topic_rank(ka: LoadFile, text: str, n: int) -> List[str]:
        assert isinstance(ka, TopicRank)
        ka.load_document(input=text[:MAX_LENGTH], language="en")
        ka.candidate_selection()
        ka.candidate_weighting()
        keyphrases: List[Tuple[str, float]] = ka.get_n_best(n=n)
        keyphrases_sorted = sorted(
            keyphrases, key=lambda kp_and_score: kp_and_score[1], reverse=True
        )
        kps = list(map(lambda kp_and_score: kp_and_score[0], keyphrases_sorted))
        return kps


class PKEAnnotator(BaseAnnotator):
    def __init__(
        self,
        top_k_words: int,
        keyphrase_approach: KeyphraseApproach,
        nprocs: int,
    ) -> None:
        self.top_k_words = top_k_words
        self.keyphrase_approach = keyphrase_approach
        self.nprocs = nprocs

    def _run(self, docs: Iterable[Document], ndocs: int) -> List[List[str]]:
        multiprocessor = Multiprocesser(self.nprocs)
        texts = map(
            lambda doc: "\n".join([chk.text for chk in doc.chunks])[:MAX_LENGTH], docs
        )
        results: List[List[str]] = multiprocessor.run(
            texts,
            func=partial(self.keyphrase_approach, n=self.top_k_words),
            desc=f"Running {self.keyphrase_approach}",
            total=ndocs,
            chunk_size=500,  # https://stackoverflow.com/questions/64515797/there-appear-to-be-6-leaked-semaphore-objects-to-clean-up-at-shutdown-warnings-w#comment126544553_65130215
        )
        return results

    def extract(self, data: LoadedData) -> Dict[str, List[str]]:
        assert data.corpus_iter_fn is not None

        kps_corpus = self._run(
            docs=data.corpus_iter_fn(), ndocs=data.meta_data["ndocs"]
        )
        doc_id2kps = {
            doc.doc_id: kps for doc, kps in zip(data.corpus_iter_fn(), kps_corpus)
        }

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
                    if doc.doc_id not in doc_id2kps:
                        leftover.append(doc)

        kps_leftover = self._run(docs=leftover, ndocs=len(leftover))
        for doc, kps in zip(leftover, kps_leftover):
            doc_id2kps[doc.doc_id] = kps

        return doc_id2kps

    def _annotate(self, data: LoadedData) -> Dict[str, str]:
        assert data.corpus_iter_fn is not None
        extract_fn = self.extract
        doc_id2kps = extract_fn(data)
        did2dsum: Dict[str, str] = {}
        for doc in data.corpus_iter_fn():
            doc_summary = " ".join(doc_id2kps[doc.doc_id])
            did2dsum[doc.doc_id] = doc_summary

        for lqs in [
            data.labeled_queries_train,
            data.labeled_queries_dev,
            data.labeled_queries_test,
        ]:
            if lqs is None:
                continue
            for lq in lqs:
                for jchk in lq.judged_chunks:
                    doc = jchk.chunk.belonging_doc
                    doc_summary = " ".join(doc_id2kps[doc.doc_id])
                    did2dsum[doc.doc_id] = doc_summary
        return did2dsum


if __name__ == "__main__":
    from dapr.hydra_schemas.dataset import NaturalQuestionsConfig

    nq = NaturalQuestionsConfig()()

    pke_summarizer = PKEAnnotator(10, KeyphraseApproach.topic_rank, 32, False)
    pke_summarizer.annotate(nq)

    doc_0: Document = next(nq.loaded_data.corpus_iter_fn())
    print(doc_0.doc_id, doc_0.title, doc_0.chunks[0].doc_summary)
