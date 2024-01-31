from functools import partial
import logging
from dapr.exps.bm25_doc_passage_hierarchy.args.dragon_plus import (
    DRAGONPlusBM25DocPassageHierarchyArguments,
)
from dapr.exps.bm25_doc_passage_hierarchy.shared_pipeline import (
    run_bm25_doc_passage_hierarchy,
)
from dapr.retrievers.dense import DRAGONPlus
from clddp.utils import is_device_zero, parse_cli, initialize_ddp, set_logger_format
from clddp.search import search

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(DRAGONPlusBM25DocPassageHierarchyArguments)
    if is_device_zero():
        args.dump_arguments()
    retriever = DRAGONPlus()
    run_bm25_doc_passage_hierarchy(
        args=args,
        passage_retriever_name="dragon_plus",
        scoped_search_function=partial(
            search,
            retriever=retriever,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
        ),
    )
