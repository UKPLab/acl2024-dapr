from dataclasses import dataclass
import re
from typing import List, Optional, Tuple, Union
from dapr.datasets.base import BaseDataset, LoadedData
from dapr.datasets.dm import Chunk, Document, JudgedChunk, LabeledQuery, Query
from dapr.utils import Multiprocesser, set_logger_format
import ir_datasets
from ir_datasets.datasets import highwire
import tqdm


@dataclass
class QRelRecord:
    query: Query
    qrel: highwire.HighwireQrel


class Genomics(BaseDataset):
    def _download(self, resource_path: str) -> None:
        pass

    def _keep_single_return(self, text: str) -> str:
        """For example,
        ```
        "\\r\\n\\r\\n\\r\\nJohn Snow and Modern-Day Environmental Epidemiology\\r\\n\\r\\n\\r\\n\\r\\nDale P. Sandler\\r\\n"
        ```
        into
        ```
        "\\r\\nJohn Snow and Modern-Day Environmental Epidemiology\\r\\nDale P. Sandler\\r\\n"
        ```
        """
        return re.sub("(\r|\n)+", "\r\n", text.strip()) + "\r\n"

    def _build_chunks(
        self, hw_doc_qrel_record: Tuple[highwire.HighwireDoc, Optional[QRelRecord]]
    ) -> Union[List[JudgedChunk], List[Chunk]]:
        hw_doc, qrel_record = hw_doc_qrel_record
        passages = [self._keep_single_return(span.text) for span in hw_doc.spans]
        marked = [False] * len(passages)
        if qrel_record is not None:
            assert qrel_record.qrel.relevance > 0
            marked = [
                True
                if (span.start, span.length)
                == (qrel_record.qrel.start, qrel_record.qrel.length)
                else False
                for span in hw_doc.spans
            ]
        input_ids = self.tokenizer(passages, add_special_tokens=False)["input_ids"]
        judged_chunks = []
        document = Document(doc_id=hw_doc.doc_id, chunks=[], title=str(hw_doc.title))
        for chunk_token_ids, mark in zip(input_ids, marked):
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
            if mark:
                judged_chunks.append(
                    JudgedChunk(
                        query=qrel_record.query,
                        chunk=chunk,
                        judgement=qrel_record.qrel.relevance,
                    )
                )

        if qrel_record is None:
            return document.chunks
        else:
            return judged_chunks

    def _build_document(self, hw_doc: highwire.HighwireDoc) -> Document:
        chunks: List[Chunk] = self._build_chunks((hw_doc, None))
        return chunks[0].belonging_doc

    def _build_corpus(self) -> List[Document]:
        dataset: ir_datasets.datasets.base.Dataset = ir_datasets.load(
            "highwire/trec-genomics-2007"
        )  # 2006 and 2007 share the same corpus
        hw_docs: ir_datasets.datasets.base._BetaPythonApiDocs = getattr(dataset, "docs")
        corpus = Multiprocesser(self.nprocs).run(
            data=hw_docs,
            func=self._build_document,
            desc="Building corpus",
            total=len(hw_docs),
        )
        return corpus

    def _build_labeled_queries(self, year: str) -> List[LabeledQuery]:
        assert year in ["2006", "2007"]
        dataset: ir_datasets.datasets.base.Dataset = ir_datasets.load(
            f"highwire/trec-genomics-{year}"
        )
        hw_docs: ir_datasets.datasets.base._BetaPythonApiDocs = getattr(dataset, "docs")
        hw_queries: ir_datasets.datasets.base._BetaPythonApiQueries = getattr(
            dataset, "queries"
        )
        hw_qrels: ir_datasets.datasets.base._BetaPythonApiQrels = getattr(
            dataset, "qrels"
        )
        hw_qrel: highwire.HighwireQrel
        labeled_queries = []
        for hw_qrel in tqdm.tqdm(
            hw_qrels, desc=f"Building labeled queries (year: {year})"
        ):
            if hw_qrel.relevance == 0:
                continue

            hw_query: ir_datasets.formats.base.GenericQuery = hw_queries.lookup(
                hw_qrel.query_id
            )
            qrel_record = QRelRecord(
                query=Query(query_id=hw_qrel.query_id, text=hw_query.text), qrel=hw_qrel
            )
            hw_doc: highwire.HighwireDoc = hw_docs.lookup(hw_qrel.doc_id)
            jchunks = self._build_chunks((hw_doc, qrel_record))
            labeled_queries.append(
                LabeledQuery(query=qrel_record.query, judged_chunks=jchunks)
            )
        return labeled_queries

    def _load_data(self, nheldout: Optional[int]) -> LoadedData:
        corpus = self._build_corpus()
        labeled_queries_2006 = self._build_labeled_queries("2006")
        labeled_queries_2007 = self._build_labeled_queries("2007")
        labeled_queries_test = []
        labeled_queries_test.extend(labeled_queries_2006)
        labeled_queries_test.extend(labeled_queries_2007)
        return LoadedData(
            corpus_iter_fn=lambda: iter(corpus),
            labeled_queries_test=labeled_queries_test,
        )


if __name__ == "__main__":
    set_logger_format()
    genomics = Genomics(resource_path="", nheldout=None, cache_loaded=True)
