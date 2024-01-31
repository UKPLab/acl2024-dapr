from typing import Dict, List
from clddp.dm import RetrievedPassageIDList, ScoredPassageID


SOFT_ZERO = 1e-4


def normalize_min_max(scores: Dict[str, float]) -> Dict[str, float]:
    if len(scores) == 0:
        return {}
    score_min = min(scores.values())
    score_max = max(scores.values())
    divisor = max(score_max - score_min, SOFT_ZERO)
    normalized = {k: (v - score_min) / divisor for k, v in scores.items()}
    return normalized


def M2C2(
    scores1: Dict[str, float],
    scores2: Dict[str, float],
    weight2: float,
) -> Dict[str, float]:
    scores1 = normalize_min_max(scores=scores1)
    scores2 = normalize_min_max(scores=scores2)
    ids_union = set(scores1) | set(scores2)
    default_score = 0
    scores = {
        k: (1 - weight2) * scores1.get(k, default_score)
        + weight2 * scores2.get(k, default_score)
        for k in ids_union
    }
    return scores


def doc_passage_fusion_with_M2C2(
    doc_results: List[RetrievedPassageIDList],
    passage_results: List[RetrievedPassageIDList],
    pid2did: Dict[str, str],
    passage_weight: float,
) -> List[RetrievedPassageIDList]:
    fused_lists = []
    for doc_result, passage_result in zip(doc_results, passage_results):
        assert doc_result.query_id == passage_result.query_id
        did2score = ScoredPassageID.build_pid2score(doc_result.scored_passage_ids)
        pid2score = ScoredPassageID.build_pid2score(passage_result.scored_passage_ids)
        doc_pid2score = {
            pid: did2score[pid2did[pid]]
            for pid in pid2score.keys()
            if pid2did[pid] in did2score
        }
        fused = M2C2(scores1=doc_pid2score, scores2=pid2score, weight2=passage_weight)
        spids = [
            ScoredPassageID(passage_id=pid, score=score) for pid, score in fused.items()
        ]
        fused_lists.append(
            RetrievedPassageIDList(
                query_id=doc_result.query_id, scored_passage_ids=spids
            )
        )
    return fused_lists
