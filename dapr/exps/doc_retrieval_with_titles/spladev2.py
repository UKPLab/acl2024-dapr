import logging
from clddp.utils import initialize_ddp, is_device_zero, parse_cli, set_logger_format
from dapr.retrievers.sparse import SPLADEv2
from dapr.exps.doc_retrieval_with_titles.shared_pipeline import (
    run_doc_retrieval_with_titles,
)
from dapr.exps.doc_retrieval_with_titles.args.spladev2 import (
    SPLADEv2lusDocRetrievalWithTitlesArguments,
)

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(SPLADEv2lusDocRetrievalWithTitlesArguments)
    retriever = SPLADEv2()
    run_doc_retrieval_with_titles(
        args=args, retriever=retriever, retriever_name="spladev2"
    )
