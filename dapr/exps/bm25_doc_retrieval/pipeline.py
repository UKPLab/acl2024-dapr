import json
import logging
import os
from clddp.dm import LabeledQuery, RetrievedPassageIDList
from clddp.evaluation import RetrievalEvaluator
from clddp.utils import is_device_zero, parse_cli, set_logger_format
from dapr.exps.bm25_doc_retrieval.args import BM25DocRetrievalArguments
from dapr.dataloader import DAPRDataConfig, DAPRDataLoader, RetrievalLevel
from dapr.retrievers.bm25 import BM25


def run_bm25_doc_retrieval() -> None:
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    args = parse_cli(BM25DocRetrievalArguments)
    args.dump_arguments()

    # Doing BM25 document retrieval:
    fdocs_ranking = os.path.join(args.output_dir, "doc_ranking_results.txt")
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
    evaluator = RetrievalEvaluator(eval_dataset=doc_dataset, split=args.split)
    eval_scores = evaluator._pytrec_eval(
        trec_scores=RetrievedPassageIDList.build_trec_scores(retrieved_docs),
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


if __name__ == "__main__":
    run_bm25_doc_retrieval()
