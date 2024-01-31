import json
import logging
import os
from typing import Iterable, List, Protocol, Set
from clddp.dm import LabeledQuery, Passage, Query, RetrievedPassageIDList
from clddp.utils import is_device_zero
from dapr.retrievers.bm25 import BM25
from dapr.exps.bm25_doc_passage_hierarchy.args.base import (
    BM25DocPassageHierarchyArguments,
)
from dapr.dataloader import DAPRDataConfig, DAPRDataLoader, RetrievalLevel
from clddp.evaluation import RetrievalEvaluator
from dapr.fusion import doc_passage_fusion_with_M2C2
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from torch import distributed as dist


class ScopedSearchFunction(Protocol):
    def __call__(
        self,
        collection_iter: Iterable[Passage],
        collection_size: int,
        queries: List[Query],
        topk: int,
        passage_scopes: List[Set[str]],  # For each query, which pids are allowed
    ) -> List[RetrievedPassageIDList]:
        pass


def run_bm25_doc_passage_hierarchy(
    args: BM25DocPassageHierarchyArguments,
    passage_retriever_name: str,
    scoped_search_function: ScopedSearchFunction,
) -> None:
    # Doing BM25 document retrieval:
    fdocs_ranking = os.path.join(args.output_dir, "doc_ranking_results.txt")
    if is_device_zero():
        doc_dataset = DAPRDataLoader(
            DAPRDataConfig(
                data_name_or_path=args.data_dir, retrieval_level=RetrievalLevel.document
            )
        ).load_data(True)
        labeled_queries = doc_dataset.get_labeled_queries(args.split)
        queries = LabeledQuery.get_unique_queries(labeled_queries)
        retriever = BM25()
        index_path = os.path.join(args.output_dir, "index")
        if not (os.path.exists(index_path) and len(os.listdir(index_path))):
            retriever.index(
                collection_iter=doc_dataset.collection_iter,
                collection_size=doc_dataset.collection_size,
                output_dir=index_path,
            )
        else:
            logging.info(f"Found existing index {index_path}")
        retrieved_docs = retriever.search(
            queries=queries,
            index_path=index_path,
            topk=args.topk,
            batch_size=args.per_device_eval_batch_size,
        )
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=retrieved_docs, fpath=fdocs_ranking, system="bm25"
        )
        logging.info(f"Saved BM25 document ranking results to {fdocs_ranking}.")
    if dist.is_initialized():
        dist.barrier()
    retrieved_docs = RetrievedPassageIDList.from_trec_csv(fdocs_ranking)

    # Search over the paragraphs which belongs to the retrieved docs:
    paragraph_dataloader = DAPRDataLoader(
        DAPRDataConfig(
            data_name_or_path=args.data_dir, retrieval_level=RetrievalLevel.paragraph
        )
    )
    paragraph_dataset = paragraph_dataloader.load_data(is_device_zero())
    pid2did = paragraph_dataloader.get_pid2did(is_device_zero())
    did2pids = paragraph_dataloader.get_did2pids(is_device_zero())
    labeled_queries = paragraph_dataset.get_labeled_queries(args.split)
    queries = LabeledQuery.get_unique_queries(labeled_queries)
    passage_scopes = [
        {pid for sdoc in qdoc.scored_passage_ids for pid in did2pids[sdoc.passage_id]}
        for qdoc in retrieved_docs
    ]
    retrieved_passages = scoped_search_function(
        collection_iter=paragraph_dataset.collection_iter,
        collection_size=paragraph_dataset.collection_size,
        queries=queries,
        topk=args.topk,
        passage_scopes=passage_scopes,
    )

    # Doing fusion:
    if is_device_zero():
        passage_weights = [
            round(weight, 1) for weight in np.arange(0, 1.1, 0.1).tolist()
        ]
        pwegiht2metrics = {}
        for passage_weight in tqdm.tqdm(passage_weights, desc="Doing fusion"):
            fused = doc_passage_fusion_with_M2C2(
                doc_results=retrieved_docs,
                passage_results=retrieved_passages,
                pid2did=pid2did,
                passage_weight=passage_weight,
            )
            evaluator = RetrievalEvaluator(
                eval_dataset=paragraph_dataset, split=args.split
            )
            report = evaluator(fused)
            pwegiht2metrics[passage_weight] = report
            freport = os.path.join(
                args.output_dir, f"metrics-pweight_{passage_weight}.json"
            )
            with open(freport, "w") as f:
                json.dump(report, f, indent=4)
            logging.info(f"Saved evaluation metrics to {freport}.")
            franked = os.path.join(
                args.output_dir, f"ranking_results-pweight_{passage_weight}.txt"
            )
            RetrievedPassageIDList.dump_trec_csv(
                retrieval_results=fused, fpath=franked, system=passage_retriever_name
            )
            logging.info(f"Saved ranking results to {franked}.")

        # Save again the metrics to be reported:
        if args.report_passage_weight:
            report_metrics = pwegiht2metrics[args.report_passage_weight]
            freport = os.path.join(args.output_dir, f"metrics.json")
            with open(freport, "w") as f:
                json.dump(report_metrics, f, indent=4)
            logging.info(f"Saved evaluation metrics to {freport}.")

        # Plot the curve:
        pweight2main_metric = {
            pweight: pwegiht2metrics[pweight][args.report_metric]
            for pweight in passage_weights
        }
        fcurve_data = os.path.join(
            args.output_dir, f"passage_weight-vs-{args.report_metric}.json"
        )
        with open(fcurve_data, "w") as f:
            json.dump(pweight2main_metric, f, indent=4)
        logging.info(f"Saved the curve data to {fcurve_data}")
        main_metrics = [pweight2main_metric[pweight] for pweight in passage_weights]
        fcurve = os.path.join(
            args.output_dir, f"passage_weight-vs-{args.report_metric}.pdf"
        )
        plt.plot(passage_weights, main_metrics)
        plt.grid(linestyle="dashed")
        plt.xlabel("passage weight")
        plt.ylabel(f"{args.report_metric}")
        plt.savefig(fcurve, bbox_inches="tight")
