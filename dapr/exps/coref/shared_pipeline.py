import json
import logging
import os
from clddp.dm import LabeledQuery, RetrievedPassageIDList
from clddp.retriever import Retriever
from clddp.utils import is_device_zero
from dapr.exps.coref.args.base import CorefArguments
from clddp.search import search
from dapr.dataloader import DAPRDataConfig, DAPRDataLoader
from clddp.evaluation import RetrievalEvaluator


def run_coref(args: CorefArguments, retriever: Retriever, retriever_name: str) -> None:
    # Actually the same as passage_only. And the difference relies on the data
    if is_device_zero():
        args.dump_arguments()
    dataset = DAPRDataLoader(DAPRDataConfig(args.data_dir)).load_data(
        progress_bar=is_device_zero()
    )
    labeled_queries = dataset.get_labeled_queries(args.split)
    queries = LabeledQuery.get_unique_queries(labeled_queries)
    retrieved = search(
        retriever=retriever,
        collection_iter=dataset.collection_iter,
        collection_size=dataset.collection_size,
        queries=queries,
        topk=args.topk,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
    )
    if is_device_zero():
        evaluator = RetrievalEvaluator(eval_dataset=dataset, split=args.split)
        report = evaluator(retrieved)
        freport = os.path.join(args.output_dir, "metrics.json")
        with open(freport, "w") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Saved evaluation metrics to {freport}.")
        franked = os.path.join(args.output_dir, "ranking_results.txt")
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=retrieved, fpath=franked, system=retriever_name
        )
        logging.info(f"Saved ranking results to {franked}.")
