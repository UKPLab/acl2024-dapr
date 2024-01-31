import json
import logging
import os
from typing import Iterable
from clddp.dm import LabeledQuery, Passage, RetrievedPassageIDList
from clddp.retriever import Retriever
from clddp.utils import is_device_zero
from dapr.dataloader import DAPRDataConfig
from dapr.exps.doc_retrieval_with_titles.args.base import (
    DocRetrievalWithTitlesArguments,
)
from clddp.search import search
from dapr.dataloader import DAPRDataConfig, DAPRDataLoader, RetrievalLevel
from clddp.evaluation import RetrievalEvaluator


def title_only_collection_iter(collection_iter: Iterable[Passage]) -> Iterable[Passage]:
    for doc in collection_iter:
        yield Passage(passage_id=doc.passage_id, text="", title=doc.title)


def run_doc_retrieval_with_titles(
    args: DocRetrievalWithTitlesArguments, retriever: Retriever, retriever_name: str
) -> None:
    # Actually the same as passage_only. And the difference relies on the data
    if is_device_zero():
        args.dump_arguments()
    dataset = DAPRDataLoader(
        DAPRDataConfig(
            data_name_or_path=args.data_dir,
            titled=True,
            retrieval_level=RetrievalLevel.document,
        )
    ).load_data(progress_bar=is_device_zero())
    labeled_queries = dataset.get_labeled_queries(args.split)
    queries = LabeledQuery.get_unique_queries(labeled_queries)
    retrieved = search(
        retriever=retriever,
        collection_iter=title_only_collection_iter(dataset.collection_iter),
        collection_size=dataset.collection_size,
        queries=queries,
        topk=args.topk,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
    )
    if is_device_zero():
        evaluator = RetrievalEvaluator(eval_dataset=dataset, split=args.split)
        eval_scores = evaluator._pytrec_eval(
            trec_scores=RetrievedPassageIDList.build_trec_scores(retrieved),
            qrels=evaluator.qrels,
        )
        report_metrics = evaluator._build_report(eval_scores)
        fdetails = os.path.join(args.output_dir, "q2d_details.jsonl")
        with open(fdetails, "w") as f:
            for qid, row in eval_scores.items():
                row = dict(eval_scores[qid])
                row["query_id"] = qid
                f.write(json.dumps(row) + "\n")
        freport = os.path.join(args.output_dir, "q2d_metrics.json")
        with open(freport, "w") as f:
            json.dump(report_metrics, f, indent=4)
        logging.info(f"Saved evaluation metrics to {freport}.")
        franked = os.path.join(args.output_dir, "doc_ranking_results.txt")
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=retrieved, fpath=franked, system=retriever_name
        )
        logging.info(f"Saved ranking results to {franked}.")
