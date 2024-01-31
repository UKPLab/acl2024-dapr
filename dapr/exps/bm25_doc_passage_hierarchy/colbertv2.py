from functools import partial
import logging
from dapr.exps.bm25_doc_passage_hierarchy.args.colbertv2 import (
    CoLBERTv2BM25DocPassageHierarchyArguments,
)
from dapr.exps.bm25_doc_passage_hierarchy.shared_pipeline import (
    run_bm25_doc_passage_hierarchy,
)
from clddp.utils import initialize_ddp, is_device_zero, parse_cli, set_logger_format
from clddp.search import search
from dapr.retrievers.late_interaction import ColBERTv2

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(CoLBERTv2BM25DocPassageHierarchyArguments)
    if is_device_zero():
        args.dump_arguments()
    retriever = ColBERTv2()
    run_bm25_doc_passage_hierarchy(
        args=args,
        passage_retriever_name="colbertv2",
        scoped_search_function=partial(
            search,
            retriever=retriever,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
        ),
    )
