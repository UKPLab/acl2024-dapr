import json
import logging
import os
from clddp.dm import LabeledQuery, RetrievedPassageIDList
from clddp.utils import is_device_zero, parse_cli, set_logger_format
from dapr.retrievers.bm25 import BM25
from dapr.exps.passage_only.args.bm25 import BM25PassageOnlyArguments
from dapr.dataloader import DAPRDataConfig, DAPRDataLoader
from clddp.evaluation import RetrievalEvaluator

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    args = parse_cli(BM25PassageOnlyArguments)
    if is_device_zero():
        args.dump_arguments()
    dataset = DAPRDataLoader(DAPRDataConfig(args.data_dir)).load_data(
        progress_bar=is_device_zero()
    )
    retriever = BM25()
    labeled_queries = dataset.get_labeled_queries(args.split)
    queries = LabeledQuery.get_unique_queries(labeled_queries)
    index_path = os.path.join(args.output_dir, "index")
    retriever.index(
        collection_iter=dataset.collection_iter,
        collection_size=dataset.collection_size,
        output_dir=index_path,
    )
    retrieved = retriever.search(
        queries=queries,
        index_path=index_path,
        topk=args.topk,
        batch_size=args.per_device_eval_batch_size,
    )
    evaluator = RetrievalEvaluator(eval_dataset=dataset, split=args.split)
    report = evaluator(retrieved)
    freport = os.path.join(args.output_dir, "metrics.json")
    with open(freport, "w") as f:
        json.dump(report, f, indent=4)
    logging.info(f"Saved evaluation metrics to {freport}.")
    franked = os.path.join(args.output_dir, "ranking_results.txt")
    RetrievedPassageIDList.dump_trec_csv(
        retrieval_results=retrieved, fpath=franked, system="retromae"
    )
    logging.info(f"Saved ranking results to {franked}.")
