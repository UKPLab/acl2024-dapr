from dataclasses import dataclass
import os
from typing import Optional, Set
from clddp.args.base import AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn
from clddp.dm import Split
from clddp.evaluation import RetrievalMetric
from clddp.utils import parse_cli


@dataclass
class BM25DocRetrievalArguments(AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn):
    data_dir: Optional[str] = None
    split: Split = Split.test
    topk: int = 1000
    per_device_eval_batch_size: int = 32
    fp16: bool = True
    report_metric: str = RetrievalMetric.ndcg_string.at(10)
    report_passage_weight: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()

    def build_output_dir(self) -> str:
        return os.path.join("exps/bm25_doc_retrieval", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(BM25DocRetrievalArguments).output_dir
    )  # For creating the logging path
