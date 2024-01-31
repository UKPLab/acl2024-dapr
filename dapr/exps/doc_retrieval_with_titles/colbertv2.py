import logging
from clddp.utils import initialize_ddp, is_device_zero, parse_cli, set_logger_format
from dapr.retrievers.late_interaction import ColBERTv2
from dapr.exps.doc_retrieval_with_titles.shared_pipeline import (
    run_doc_retrieval_with_titles,
)
from dapr.exps.doc_retrieval_with_titles.args.colbertv2 import (
    ColBERTv2DocRetrievalWithTitlesArguments,
)

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(ColBERTv2DocRetrievalWithTitlesArguments)
    retriever = ColBERTv2(
        query_max_length=args.query_max_length,
        passage_max_length=args.passage_max_length,
    )  # titles are extra short
    run_doc_retrieval_with_titles(
        args=args, retriever=retriever, retriever_name="colbertv2"
    )
