from itertools import chain
from typing import Dict, List
from dapr.annotators.base import BaseAnnotator
from dapr.datasets.base import BaseDataset
from dapr.datasets.dm import Document
from nltk import sent_tokenize
import nltk
import tqdm


class LeadingSentencesAnnotator(BaseAnnotator):
    def __init__(self, nlead: int) -> None:
        self.nlead = nlead

    def extract_leading_sentences(self, doc: Document) -> List[str]:
        leading_sentences = []
        for chunk in doc.chunks:
            leading_sentences.extend(sent_tokenize(chunk.text))
            if len(leading_sentences) >= self.nlead:
                break
        return leading_sentences[: self.nlead]

    def extract_doc_summary(self, doc: Document) -> str:
        return " ".join(self.extract_leading_sentences(doc))

    def _annotate(self, dataset: BaseDataset) -> Dict[str, str]:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

        data = dataset.loaded_data
        assert data.corpus_iter_fn is not None

        did2dsum: Dict[str, str] = {}
        for doc in tqdm.tqdm(
            data.corpus_iter_fn(),
            total=data.meta_data["ndocs"],
            desc="Annotating corpus",
        ):
            doc_summary = self.extract_doc_summary(doc)
            did2dsum[doc.doc_id] = doc_summary

        for lqs in [
            data.labeled_queries_dev,
            data.labeled_queries_test,
            data.labeled_queries_train,
        ]:
            if lqs is None:
                continue
            for lq in tqdm.tqdm(lqs, desc="Annotating labeled queries"):
                for jchk in lq.judged_chunks:
                    doc = jchk.chunk.belonging_doc
                    doc_summary = self.extract_doc_summary(doc)
                    did2dsum[doc.doc_id] = doc_summary
        return did2dsum
