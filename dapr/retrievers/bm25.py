from __future__ import annotations
from dataclasses import dataclass
import json
import os
import shutil
from typing import Callable, Dict, Iterable, List, Type
import tqdm
from clddp.dm import Passage, Query, RetrievedPassageIDList, ScoredPassageID
import logging


@dataclass
class PyseriniHit:
    docid: str
    score: float

    @classmethod
    def from_pyserini(cls: Type[PyseriniHit], hit: PyseriniHit) -> PyseriniHit:
        """This conversion step makes the memory releasable (otherwise leaked by Java)."""
        return PyseriniHit(docid=hit.docid, score=hit.score)


@dataclass
class PyseriniCollectionRow:
    id: str
    title: str
    contents: str

    def to_json(self) -> Dict[str, str]:
        return {"id": self.id, "title": self.title, "contents": self.contents}


class BM25:
    CORPUS_FOLDER = "corpus"

    def __init__(self) -> None:
        os.environ["_JAVA_OPTIONS"] = (
            "-Xmx5g"  # Otherwise it would cause to huge memory leak!
        )

    def index(
        self,
        collection_iter: Iterable[Passage],
        collection_size: int,
        output_dir: str,
        nthreads: int = 12,
        keep_converted_corpus: bool = False,
    ) -> None:
        logging.info("Run indexing.")
        import pyserini.index.lucene
        from jnius import autoclass

        # Converting into the required format for indexing:
        coprus_path = os.path.join(output_dir, self.CORPUS_FOLDER)
        os.makedirs(coprus_path, exist_ok=True)
        with open(os.path.join(coprus_path, "texts.jsonl"), "w") as f:
            for psg in tqdm.tqdm(
                collection_iter,
                total=collection_size,
                desc="Converting to pyserini format",
            ):
                title = psg.title if psg.title else ""
                json_line = PyseriniCollectionRow(
                    id=psg.passage_id, title=title, contents=psg.text
                ).to_json()
                f.write(json.dumps(json_line) + "\n")

        # Run actual indexing:
        args = [
            "-collection",
            "JsonCollection",
            "-generator",
            "DefaultLuceneDocumentGenerator",
            "-threads",
            str(nthreads),
            "-input",
            coprus_path,
            "-index",
            output_dir,
            # "-storeRaw",
            "-storePositions",
            "-storeDocvectors",
            "-fields",
            "title",
        ]
        JIndexCollection = autoclass("io.anserini.index.IndexCollection")
        index_fn: Callable[
            [
                List[str],
            ],
            None,
        ] = getattr(JIndexCollection, "main")
        try:
            index_fn(args)
        except Exception as e:
            shutil.rmtree(output_dir)
            raise e
        logging.info(f"Done indexing. Index path: {output_dir}")
        if not keep_converted_corpus:
            logging.info(f"Removing the converted corpus {coprus_path}")
            shutil.rmtree(coprus_path)

    def search(
        self,
        queries: List[Query],
        index_path: str,
        topk: int,
        batch_size: int,
        contents_weight: float = 1.0,
        title_weight: float = 1.0,
        nthreads: int = 12,
    ) -> List[RetrievedPassageIDList]:
        from pyserini.search import SimpleSearcher

        # Actual search:
        fields = {
            "contents": contents_weight,
            "title": title_weight,
        }
        searcher = SimpleSearcher(index_path)
        searcher.set_bm25()
        qid2hits_all: Dict[str, List[PyseriniHit]] = {}
        for b in tqdm.trange(0, len(queries), batch_size, desc="Query batch"):
            e = b + batch_size
            qid2hits = searcher.batch_search(
                queries=list(map(lambda query: query.text, queries[b:e])),
                qids=list(map(lambda query: query.query_id, queries[b:e])),
                k=topk,
                threads=nthreads,
                fields=fields,
            )
            qid2hits_converted = {
                qid: list(map(PyseriniHit.from_pyserini, hits))
                for qid, hits in qid2hits.items()
            }
            qid2hits_all.update(qid2hits_converted)
        searcher.close()

        # Convert to the clddp format:
        retrieved = []
        for query in queries:
            hits = qid2hits_all[query.query_id]
            spids = [
                ScoredPassageID(passage_id=hit.docid, score=hit.score) for hit in hits
            ]
            retrieved.append(
                RetrievedPassageIDList(
                    query_id=query.query_id, scored_passage_ids=spids
                )
            )
        return retrieved
