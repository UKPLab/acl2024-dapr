import logging
from clddp.utils import initialize_ddp, is_device_zero, parse_cli, set_logger_format
from dapr.retrievers.dense import DRAGONPlus
from dapr.exps.doc_retrieval_with_titles.shared_pipeline import (
    run_doc_retrieval_with_titles,
)
from dapr.exps.doc_retrieval_with_titles.args.dragon_plus import (
    DRAGONPlusDocRetrievalWithTitlesArguments,
)

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(DRAGONPlusDocRetrievalWithTitlesArguments)
    retriever = DRAGONPlus()
    run_doc_retrieval_with_titles(
        args=args, retriever=retriever, retriever_name="dragon_plus"
    )
