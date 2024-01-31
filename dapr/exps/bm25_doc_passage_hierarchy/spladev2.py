from functools import partial
import logging
from dapr.exps.bm25_doc_passage_hierarchy.args.spladev2 import (
    SPLADEv2BM25DocPassageHierarchyArguments,
)
from dapr.exps.bm25_doc_passage_hierarchy.shared_pipeline import (
    run_bm25_doc_passage_hierarchy,
)
from dapr.retrievers.sparse import SPLADEv2
from clddp.utils import is_device_zero, parse_cli, initialize_ddp, set_logger_format
from clddp.search import search

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(SPLADEv2BM25DocPassageHierarchyArguments)
    if is_device_zero():
        args.dump_arguments()
    retriever = SPLADEv2()
    run_bm25_doc_passage_hierarchy(
        args=args,
        passage_retriever_name="spladev2",
        scoped_search_function=partial(
            search,
            retriever=retriever,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            fp16=args.fp16,
        ),
    )
