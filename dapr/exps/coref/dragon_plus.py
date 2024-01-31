import logging
from clddp.utils import initialize_ddp, is_device_zero, parse_cli, set_logger_format
from dapr.retrievers.dense import DRAGONPlus
from dapr.exps.coref.shared_pipeline import run_coref
from dapr.exps.coref.args.dragon_plus import DRAGONPlusCorefArguments

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(DRAGONPlusCorefArguments)
    retriever = DRAGONPlus()
    run_coref(args=args, retriever=retriever, retriever_name="dragon_plus")
