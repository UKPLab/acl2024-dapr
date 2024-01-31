import logging
from clddp.utils import initialize_ddp, is_device_zero, parse_cli, set_logger_format
from dapr.retrievers.sparse import SPLADEv2
from dapr.exps.coref.shared_pipeline import run_coref
from dapr.exps.coref.args.spladev2 import SPLADEvCorefArguments

if __name__ == "__main__":
    set_logger_format(logging.INFO if is_device_zero() else logging.WARNING)
    initialize_ddp()
    args = parse_cli(SPLADEvCorefArguments)
    retriever = SPLADEv2()
    run_coref(args=args, retriever=retriever, retriever_name="spladev2")
