from typing import Dict, List
import numpy as np

import pandas as pd
from dapr.datasets.dm import LabeledQuery
from dapr.models.dm import RetrievedChunkList
import tqdm


def build_results_table(
    retrieval_results: List[RetrievedChunkList],
    labeled_queries: List[LabeledQuery],
    metric: str,
    evaluation_scores: List[float],
    top_k: int = 20,
    precision: int = 4,
) -> pd.DataFrame:
    """Build a pandas table for the retrieval results."""
    columns = [
        "qid",
        "query",
        "doc_id",
        "doc_title",
        "chunk_id",
        "chunk",
        "rank",
        "judgement",
        metric,
        "score",
    ]

    data = []
    qid2lqs = LabeledQuery.group_by_qid(labeled_queries)
    for rr, es in tqdm.tqdm(
        zip(retrieval_results, evaluation_scores),
        desc="Building result table",
        total=len(retrieval_results),
    ):
        query = rr.query
        if not rr.descending:
            rr = rr.sorted()

        # First append the gold chunk(s):
        cid2judgement = LabeledQuery.build_cid2judgement(labeled_queries)
        for lq in qid2lqs[query.query_id]:
            for jchk in lq.judged_chunks:
                row = {
                    "qid": query.query_id,
                    "query": query.text,
                    "doc_id": jchk.chunk.belonging_doc.doc_id,
                    "doc_title": jchk.chunk.belonging_doc.title,
                    "chunk_id": jchk.chunk.chunk_id,
                    "chunk": jchk.chunk.text,
                    "rank": -1,
                    "judgement": jchk.judgement,
                    metric: es,
                    "score": -1,
                }
                data.append(row)

        # Then the retrieved chunks:
        for i, schk in enumerate(rr.scored_chunks):
            if i >= top_k:
                continue

            row = {
                "qid": query.query_id,
                "query": query.text,
                "doc_id": schk.chunk.belonging_doc.doc_id,
                "doc_title": schk.chunk.belonging_doc.title,
                "chunk_id": schk.chunk.chunk_id,
                "chunk": schk.chunk.text,
                "rank": i + 1,
                "judgement": cid2judgement.get(schk.chunk.chunk_id, np.NaN),
                metric: es,
                "score": round(schk.score, precision),
            }
            data.append(row)

    return pd.DataFrame(data, columns=columns)
