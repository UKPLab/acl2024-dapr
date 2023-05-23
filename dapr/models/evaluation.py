from dataclasses import dataclass
from enum import Enum
import json
import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

import pytrec_eval

from dapr.models.dm import RetrievalLevel, RetrievedChunkList, RetrievedDocumentList

from dapr.datasets.dm import Document, Query, Split, LabeledQuery, LoadedData


class RetrievalMetric(str, Enum):
    map_string = "map_cut"
    ndcg_string = "ndcg_cut"
    recall_string = "recall"
    precision_string = "P"
    rr_string = "recip_rank"

    @staticmethod
    def cutoff(metric_string: str) -> int:
        return int(metric_string.split("_")[-1])

    def at(self, k: int) -> str:
        return f"{self}_{k}"

    def trec_string(self, k_values: Tuple[int]) -> str:
        if self is RetrievalMetric.rr_string:
            return self

        return f"{self}." + ",".join([str(k) for k in k_values])


@dataclass
class EvaluationOutput:
    summary: Dict[str, float]
    details: Dict[str, Dict[str, float]]  # qid -> metric -> eval. score
    level: RetrievalLevel
    report_prefix: str
    trec_scores: Dict[str, Dict[str, float]]  # qid -> pid/did -> ranking score


class LongDocumentEvaluator:
    """Do evaluation on both chunk-level and document-level."""

    def __init__(
        self,
        data: LoadedData,
        results_dir: str,
        split: Split,
        min_plabel: int,
        metrics: Tuple[str] = (
            RetrievalMetric.ndcg_string.at(10),
            RetrievalMetric.rr_string.at(10),
            RetrievalMetric.recall_string.at(100),
        ),
        main_metric: str = RetrievalMetric.ndcg_string.at(10),
        topk: int = 1000,
        precision: int = 4,
    ) -> None:
        self.data = data
        self.results_dir = results_dir
        self.split = split
        self.min_plabel = min_plabel
        self.metrics = list(
            filter(lambda m: RetrievalMetric.cutoff(m) <= topk, metrics)
        )
        assert main_metric in metrics
        self.main_metric = main_metric
        self.topk = topk
        self.precision = precision
        self.labeled_queries = {
            Split.dev: data.labeled_queries_dev,
            Split.test: data.labeled_queries_test,
        }[split]
        self.queries: List[Query] = LabeledQuery.get_unique_queries(
            self.labeled_queries
        )
        self.qrels = LabeledQuery.build_qrels(self.labeled_queries)
        self.qrels_doc = LabeledQuery.build_qrels_doc(self.labeled_queries)
        self.ndocs = data.meta_data["ndocs"]
        self.nchunks = data.meta_data["nchunks"]
        self.pool_identifier = data.meta_data["corpus_identifier"]
        self.logger = logging.getLogger(__name__)

    @property
    def pool(self) -> Iterable[Document]:
        return self.data.corpus_iter_fn()

    def _post_process_rr(self, eval_scores: Dict[str, Dict[str, float]]):
        """pytrec_eval does not support MRR at K originally."""
        rr_cutoffs = [
            int(m.split("_")[-1])
            for m in self.metrics
            if RetrievalMetric.rr_string in m
        ]
        for k in rr_cutoffs:
            min_rr = 1 / k
            for metric2score in eval_scores.values():
                score = metric2score[RetrievalMetric.rr_string]
                if score < min_rr:
                    score = 0
                metric2score[RetrievalMetric.rr_string.at(k)] = score

    def _pytrec_eval(
        self, trec_scores: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        self.logger.info("Running pytrec-eval")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, self.metrics)
        eval_scores: Dict[str, Dict[str, float]] = evaluator.evaluate(trec_scores)
        self._post_process_rr(eval_scores)
        return eval_scores

    def _build_report(
        self, eval_scores: Dict[str, Dict[str, float]], prefix: str = ""
    ) -> Dict[str, float]:
        report: Dict[str, float] = {}
        for metric in self.metrics:
            item_name = f"{prefix}{metric}"
            report[item_name] = round(
                np.mean([m2s[metric] for m2s in eval_scores.values()]).tolist(),
                self.precision,
            )
        report = dict(sorted(report.items(), key=lambda kv: kv[0]))
        return report

    def __call__(
        self,
        retrieved: Union[List[RetrievedChunkList], List[RetrievedDocumentList]],
        level: RetrievalLevel,
        report_prefix: str = "",
    ) -> EvaluationOutput:
        if level is RetrievalLevel.chunk:
            retrieved: List[RetrievedChunkList]
            trec_scores = RetrievedChunkList.build_trec_scores(retrieved)
            qrels = self.qrels
            level_prefix = "Q2C"
        else:
            assert level is RetrievalLevel.document
            retrieved: List[RetrievedDocumentList]
            trec_scores = RetrievedDocumentList.build_trec_scores(retrieved)
            qrels = self.qrels_doc
            level_prefix = "Q2D"

        eval_scores = self._pytrec_eval(trec_scores=trec_scores, qrels=qrels)
        prefix = "/".join([level_prefix, report_prefix]) + "/"
        report = self._build_report(eval_scores=eval_scores, prefix=prefix)
        logging.info(
            f"Evaluation results ({self.split}):\n{json.dumps(report, indent=4)}"
        )
        return EvaluationOutput(
            summary=report,
            details=eval_scores,
            level=level,
            report_prefix=report_prefix,
            trec_scores=trec_scores,
        )
