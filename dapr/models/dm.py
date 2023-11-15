from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Type, TypedDict, Optional

from dapr.datasets.dm import Chunk, Document, Query
from dapr.models.encoding import SimilarityFunction
from dapr.utils import SOFT_ZERO
import numpy as np
from transformers import AutoTokenizer
import tqdm


class ScoredChunkJson(TypedDict):
    chunk_id: str
    doc_id: str
    score: float


@dataclass
class ScoredChunk:
    """A chunk with this relevance score."""

    chunk_id: str
    doc_id: str  # Its belonging document
    score: float

    @staticmethod
    def sort(scored_chunks: List[ScoredChunk]) -> List[ScoredChunk]:
        """Sort the passages in a descending order."""
        return list(
            sorted(
                scored_chunks,
                key=lambda schk: schk.score,
                reverse=True,
            )
        )

    @staticmethod
    def convert_to_dict(schks: List[ScoredChunk]) -> Dict[str, float]:
        return {schk.chunk_id: schk.score for schk in schks}

    def to_json(self) -> ScoredChunkJson:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "score": self.score}


class ScoredDocumentJson(TypedDict):
    doc_id: str
    score: float


@dataclass
class ScoredDocument:
    """A document with this relevance score."""

    doc_id: str
    score: float

    @staticmethod
    def sort(scored_documents: List[ScoredDocument]) -> List[ScoredDocument]:
        """Sort the documents in a descending order."""
        return list(
            sorted(
                scored_documents,
                key=lambda sdoc: sdoc.score,
                reverse=True,
            )
        )

    @staticmethod
    def convert_to_dict(sdocs: List[ScoredDocument]) -> Dict[str, float]:
        return {sdoc.doc_id: sdoc.score for sdoc in sdocs}

    def to_json(self) -> ScoredDocumentJson:
        return {"doc_id": self.doc_id, "score": self.score}


class RetrievedChunkListJson(TypedDict):
    query_id: str
    scored_chunks: List[ScoredChunkJson]


@dataclass
class RetrievedChunkList:
    """For a given query, what are the scored chunks by retrieval."""

    query_id: str
    scored_chunks: List[ScoredChunk]
    descending: bool = False

    def sorted(self) -> RetrievedChunkList:
        """Sort the scored schunks in the retrieval result."""
        return RetrievedChunkList(
            query_id=self.query_id,
            scored_chunks=ScoredChunk.sort(self.scored_chunks),
            descending=True,
        )

    @staticmethod
    def sort_all(
        retrieved: List[RetrievedChunkList],
    ) -> List[RetrievedChunkList]:
        """Sort the passages in each of the retrieval result."""
        return [rcl.sorted() for rcl in retrieved]

    @staticmethod
    def build_trec_scores(
        retrieved: List[RetrievedChunkList],
    ) -> Dict[str, Dict[str, float]]:
        trec_scores = {
            rcl.query_id: {schk.chunk_id: schk.score for schk in rcl.scored_chunks}
            for rcl in retrieved
        }
        return trec_scores

    def max_p(self) -> RetrievedDocumentList:
        """MaxP: Convert the RetrievedChunkList into RetrievedDocumentList by keep the scored chunk with the highest score."""
        did2score = {}
        for schk in self.scored_chunks:
            doc_id = schk.doc_id
            did2score.setdefault(doc_id, schk.score)
            did2score[doc_id] = max(
                did2score[doc_id], schk.score
            )  # MaxP: Keep only the highest score over passages/chunks
        rdl = RetrievedDocumentList(
            query_id=self.query_id,
            scored_documents=[
                ScoredDocument(doc_id=did, score=did2score[did])
                for did, score in did2score.items()
            ],
        )
        return rdl

    @classmethod
    def from_scores(
        cls: Type[RetrievedChunkList],
        scores: Dict[str, float],
        query_id: str,
        cid2did: Dict[str, str],
    ) -> RetrievedChunkList:
        scored_chunks = [
            ScoredChunk(chunk_id=cid, doc_id=cid2did[cid], score=score)
            for cid, score in scores.items()
        ]
        rcl = cls(query_id=query_id, scored_chunks=scored_chunks)
        return rcl

    def to_json(self) -> RetrievedChunkListJson:
        return {
            "query_id": self.query_id,
            "scored_chunks": [schk.to_json() for schk in self.scored_chunks],
        }

    @classmethod
    def from_json(
        cls: Type[RetrievedChunkList], rcl_json: RetrievedChunkListJson
    ) -> RetrievedChunkList:
        scored_chunks = []
        for schk_json in rcl_json["scored_chunks"]:
            scored_chunks.append(
                ScoredChunk(
                    chunk_id=schk_json["chunk_id"],
                    doc_id=schk_json["doc_id"],
                    score=schk_json["score"],
                )
            )
        return cls(query_id=rcl_json["query_id"], scored_chunks=scored_chunks)


class RetrievedDocumentListJson(TypedDict):
    query_id: str
    scored_documents: List[ScoredDocumentJson]


@dataclass
class RetrievedDocumentList:
    """For a given query, what are the scored documents by retrieval."""

    query_id: str
    scored_documents: List[ScoredDocument]
    descending: bool = False

    def sorted(self) -> RetrievedDocumentList:
        """Sort the scored documents in the retrieval result."""
        return RetrievedDocumentList(
            query_id=self.query_id,
            scored_documents=ScoredDocument.sort(self.scored_documents),
            descending=True,
        )

    @staticmethod
    def sort_all(
        retrieved: List[RetrievedDocumentList],
    ) -> List[RetrievedDocumentList]:
        """Sort the documents in each of the retrieval result."""
        return [rdl.sorted() for rdl in retrieved]

    @staticmethod
    def build_trec_scores(
        retrieved: List[RetrievedDocumentList],
    ) -> Dict[str, Dict[str, float]]:
        trec_scores = {
            rdl.query_id: {sdoc.doc_id: sdoc.score for sdoc in rdl.scored_documents}
            for rdl in retrieved
        }
        return trec_scores

    @classmethod
    def from_scores(
        cls: Type[RetrievedDocumentList],
        scores: Dict[str, float],
        query_id: str,
    ) -> RetrievedDocumentList:
        scored_documents = [
            ScoredDocument(doc_id=did, score=score) for did, score in scores.items()
        ]
        rcl = cls(query_id=query_id, scored_documents=scored_documents)
        return rcl

    def to_json(self) -> RetrievedDocumentListJson:
        return {
            "query_id": self.query_id,
            "scored_documents": [sdoc.to_json() for sdoc in self.scored_documents],
        }

    @classmethod
    def from_json(
        cls: Type[RetrievedDocumentList], rdl_json: RetrievedDocumentListJson
    ) -> RetrievedDocumentList:
        scored_documents = []
        for sdoc_json in rdl_json["scored_documents"]:
            scored_documents.append(
                ScoredDocument(doc_id=sdoc_json["doc_id"], score=sdoc_json["score"])
            )
        return cls(query_id=rdl_json["query_id"], scored_documents=scored_documents)

    @classmethod
    def from_rcl(
        cls: Type[RetrievedDocumentList], rcl: RetrievedChunkList
    ) -> RetrievedDocumentList:
        rdl = cls(query_id=rcl.query_id, scored_documents=[])
        did2score = {}  # Keep only the best
        for schk in rcl.scored_chunks:
            if schk.score > did2score.get(schk.doc_id, 0):
                did2score[schk.doc_id] = schk.score
        for did, score in did2score.items():
            sdoc = ScoredDocument(doc_id=did, score=score)
            rdl.scored_documents.append(sdoc)
        return rdl


def normalize_min_max(scores: Dict[str, float]) -> Dict[str, float]:
    if len(scores) == 0:
        return {}
    score_min = min(scores.values())
    score_max = max(scores.values())
    divisor = max(score_max - score_min, SOFT_ZERO)
    normalized = {k: (v - score_min) / divisor for k, v in scores.items()}
    return normalized


def normalize_theoretical_minimum(
    scores: Dict[str, float], theoretical_minimum: float
) -> Dict[str, float]:
    if len(scores) == 0:
        return {}
    score_max = max(scores.values())
    divisor = max(score_max - theoretical_minimum, SOFT_ZERO)
    normalized = {k: (v - theoretical_minimum) / divisor for k, v in scores.items()}
    return normalized


def M2C2(
    scores1: Dict[str, float],
    scores2: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    scores1 = normalize_min_max(scores=scores1)
    scores2 = normalize_min_max(scores=scores2)
    ids_union = set(scores1) | set(scores2)
    default_score = 0
    scores = {
        k: alpha * scores1.get(k, default_score)
        + (1 - alpha) * scores2.get(k, default_score)
        for k in ids_union
    }
    return scores


def TM2C2(
    scores1: Dict[str, float],
    scores2: Dict[str, float],
    alpha: float,
    similarity_function1: SimilarityFunction,
    similarity_function2: SimilarityFunction,
    query_tokens: int,
) -> Dict[str, float]:
    sf2tm = {
        SimilarityFunction.dot_product: 0,
        SimilarityFunction.cos_sim: -1,
        SimilarityFunction.maxsim: -query_tokens,
    }
    default_score1 = sf2tm[similarity_function1]
    default_score2 = sf2tm[similarity_function2]
    scores1 = normalize_theoretical_minimum(
        scores=scores1, theoretical_minimum=default_score1
    )
    scores2 = normalize_theoretical_minimum(
        scores=scores2, theoretical_minimum=default_score2
    )
    ids_union = set(scores1) | set(scores2)
    scores = {
        k: alpha * scores1.get(k, default_score1)
        + (1 - alpha) * scores2.get(k, default_score2)
        for k in ids_union
    }
    return scores


def RRF(
    scores1: Dict[str, float], scores2: Dict[str, float], eta: float = 60
) -> Dict[str, float]:
    ranking1 = {
        k: i
        for i, (k, _) in enumerate(
            sorted(scores1.items(), key=lambda kv: kv[1], reverse=True)
        )
    }
    ranking2 = {
        k: i
        for i, (k, _) in enumerate(
            sorted(scores2.items(), key=lambda kv: kv[1], reverse=True)
        )
    }
    ids_union = set(scores1) | set(scores2)
    default_ranking = 1e8
    scores = {
        k: 1 / (eta + ranking1.get(k, default_ranking))
        + 1 / (eta + ranking2.get(k, default_ranking))
        for k in ids_union
    }
    return scores


def RerankingMerge(
    scores_tar: Dict[str, float], scores_ref: Dict[str, float]
) -> Dict[str, float]:
    """Rerank `scores_tar` wrt `scores_ref`."""
    # Reverse=False since higher score means more relevance
    default_score = 0
    reranked = {
        k: i
        for i, (k, _) in enumerate(
            sorted(
                scores_tar.items(),
                key=lambda kv: scores_ref.get(kv[0], default_score),
                reverse=False,
            )
        )
    }
    return reranked


class RetrievalResultCombination(str, Enum):
    m2c2 = "m2c2"
    tm2c2 = "tm2c2"
    rrf = "rrf"
    rm12 = "rm12"
    rm21 = "rm21"

    def __call__(
        self,
        scores1: Dict[str, float],
        scores2: Dict[str, float],
        similarity_function1: SimilarityFunction,
        similarity_function2: SimilarityFunction,
        name1: str,
        name2: str,
        topk: int,
        alpha: Optional[float] = None,
        query_tokens: Optional[int] = None,
    ) -> Tuple[str, Dict[str, float]]:
        method: Optional[str] = None
        scores: Optional[Dict[str, float]] = None
        if self == RetrievalResultCombination.m2c2:
            assert alpha is not None
            scores = M2C2(scores1=scores1, scores2=scores2, alpha=alpha)
            weight_retriever_pairs = sorted(
                [(str(round(alpha, 1)), name1), (str(round(1 - alpha, 1)), name2)],
                key=lambda kv: kv[1],
            )
            weight_retriever_pairs_label = "_".join(
                ["_".join(kv) for kv in weight_retriever_pairs]
            )
            method = f"M2C2-{weight_retriever_pairs_label}"
        elif self == RetrievalResultCombination.tm2c2:
            assert alpha is not None
            assert query_tokens is not None
            scores = TM2C2(
                scores1=scores1,
                scores2=scores2,
                alpha=alpha,
                similarity_function1=similarity_function1,
                similarity_function2=similarity_function2,
                query_tokens=query_tokens,
            )
            weight_retriever_pairs = sorted(
                [(str(round(alpha, 1)), name1), (str(round(1 - alpha, 1)), name2)],
                key=lambda kv: kv[1],
            )
            weight_retriever_pairs_label = "_".join(
                ["_".join(kv) for kv in weight_retriever_pairs]
            )
            method = f"TM2C2-{weight_retriever_pairs_label}"
        elif self == RetrievalResultCombination.rrf:
            scores = RRF(scores1=scores1, scores2=scores2)
            method = "RRF"
        elif self == RetrievalResultCombination.rm12:
            scores = RerankingMerge(scores_tar=scores1, scores_ref=scores2)
            method = f"RM-{name1}_wrt_{name2}"
        elif self == RetrievalResultCombination.rm21:
            scores = RerankingMerge(scores_tar=scores2, scores_ref=scores1)
            method = f"RM-{name2}_wrt_{name1}"
        else:
            raise NotImplementedError

        scores: Dict[str, float]
        if len(scores) > topk:
            scores = dict(
                sorted(scores.items(), key=lambda id_score: id_score[1], reverse=True)[
                    :topk
                ]
            )
        return method, scores


def rcls_x_rcls(
    queries: List[Query],
    rcls1: List[RetrievedChunkList],
    rcls2: List[RetrievedChunkList],
    similarity_function1: SimilarityFunction,
    similarity_function2: SimilarityFunction,
    name1: str,
    name2: str,
    topk: int,
) -> Iterable[Tuple[str, List[RetrievedChunkList]]]:
    """Gives actually union of retrieved chunks."""
    assert len(rcls1) == len(rcls2)
    qid2query = {query.query_id: query for query in queries}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cid2did = {}
    for rcl in rcls1 + rcls2:
        for schk in rcl.scored_chunks:
            cid2did[schk.chunk_id] = schk.doc_id
    for rrc in RetrievalResultCombination:
        alphas: List[Optional[float]] = []
        if rrc in [RetrievalResultCombination.m2c2, RetrievalResultCombination.tm2c2]:
            alphas = np.arange(0, 1.1, 0.1).tolist()
        else:
            alphas = [None]
        for alpha in alphas:
            rcls = []
            for rcl1, rcl2 in tqdm.tqdm(
                zip(rcls1, rcls2),
                desc=f"RCLs x RCLs -> RCLs ({rrc}, {alpha})",
                total=len(rcls1),
            ):
                assert rcl1.query_id == rcl2.query_id
                query_id = rcl1.query_id
                ntokens = len(tokenizer.tokenize(qid2query[query_id].text))
                method, scores = rrc(
                    scores1=ScoredChunk.convert_to_dict(rcl1.scored_chunks),
                    scores2=ScoredChunk.convert_to_dict(rcl2.scored_chunks),
                    similarity_function1=similarity_function1,
                    similarity_function2=similarity_function2,
                    name1=name1,
                    name2=name2,
                    topk=topk,
                    alpha=alpha,
                    query_tokens=ntokens,
                )
                rcl = RetrievedChunkList.from_scores(
                    scores=scores, query_id=query_id, cid2did=cid2did
                )
                rcls.append(rcl)
            yield method, rcls


def rdls_x_rdls(
    queries: List[Query],
    rdls1: List[RetrievedDocumentList],
    rdls2: List[RetrievedDocumentList],
    similarity_function1: SimilarityFunction,
    similarity_function2: SimilarityFunction,
    name1: str,
    name2: str,
    topk: int,
) -> Iterable[Tuple[str, List[RetrievedDocumentList]]]:
    """Gives actually union of retrieved documents."""
    assert len(rdls1) == len(rdls2)
    qid2query = {query.query_id: query for query in queries}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for rrc in RetrievalResultCombination:
        alphas: List[Optional[float]] = []
        if rrc in [RetrievalResultCombination.m2c2, RetrievalResultCombination.tm2c2]:
            alphas = np.arange(0, 1.1, 0.1).tolist()
        else:
            alphas = [None]
        for alpha in alphas:
            rdls = []
            for rdl1, rdl2 in tqdm.tqdm(
                zip(rdls1, rdls2),
                desc=f"RDLs x RDLs -> RDLs ({rrc}, {alpha})",
                total=len(rdls1),
            ):
                assert rdl1.query_id == rdl2.query_id
                query_id = rdl1.query_id
                ntokens = len(tokenizer.tokenize(qid2query[query_id].text))
                method, scores = rrc(
                    scores1=ScoredDocument.convert_to_dict(rdl1.scored_documents),
                    scores2=ScoredDocument.convert_to_dict(rdl2.scored_documents),
                    similarity_function1=similarity_function1,
                    similarity_function2=similarity_function2,
                    name1=name1,
                    name2=name2,
                    topk=topk,
                    alpha=alpha,
                    query_tokens=ntokens,
                )
                rdl = RetrievedDocumentList.from_scores(
                    scores=scores, query_id=query_id
                )
                rdls.append(rdl)
            yield method, rdls


def rcls_x_rdls(
    queries: List[Query],
    rcls: List[RetrievedChunkList],
    rdls: List[RetrievedDocumentList],
    similarity_function1: SimilarityFunction,
    similarity_function2: SimilarityFunction,
    name1: str,
    name2: str,
    topk: int,
) -> Iterable[Tuple[str, List[RetrievedChunkList]]]:
    assert len(rdls) == len(rcls)
    qid2query = {query.query_id: query for query in queries}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    cid2did = {}
    for rcl in rcls:
        for schk in rcl.scored_chunks:
            cid2did[schk.chunk_id] = schk.doc_id
    for rrc in RetrievalResultCombination:
        if rrc is RetrievalResultCombination.rm21:
            # In this case, we cannot rerank RDLs wrt RCLs
            continue

        alphas: List[Optional[float]] = []
        if rrc in [RetrievalResultCombination.m2c2, RetrievalResultCombination.tm2c2]:
            alphas = np.arange(0, 1.1, 0.1).tolist()
        else:
            alphas = [None]
        for alpha in alphas:
            rcls_fused = []
            for rcl, rdl in tqdm.tqdm(
                zip(rcls, rdls),
                desc=f"RCLs x RDLs -> RCLs ({rrc}, {alpha})",
                total=len(rcls),
            ):
                assert rcl.query_id == rdl.query_id
                query_id = rcl.query_id
                ntokens = len(tokenizer.tokenize(qid2query[query_id].text))
                scores_rdl = ScoredDocument.convert_to_dict(rdl.scored_documents)
                scores2 = {
                    schk.chunk_id: scores_rdl[schk.doc_id]
                    for schk in rcl.scored_chunks
                    if schk.doc_id in scores_rdl
                }
                method, scores = rrc(
                    scores1=ScoredChunk.convert_to_dict(rcl.scored_chunks),
                    scores2=scores2,
                    similarity_function1=similarity_function1,
                    similarity_function2=similarity_function2,
                    name1=name1,
                    name2=name2,
                    topk=topk,
                    alpha=alpha,
                    query_tokens=ntokens,
                )
                rcl = RetrievedChunkList.from_scores(
                    scores=scores, query_id=query_id, cid2did=cid2did
                )
                rcls_fused.append(rcl)
            yield method, rcls_fused


class RetrievalLevel(str, Enum):
    chunk = "chunk"
    document = "document"
