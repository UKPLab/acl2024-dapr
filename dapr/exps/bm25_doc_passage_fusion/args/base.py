from dataclasses import dataclass
from typing import Optional, Set
from clddp.args.base import AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn
from clddp.dm import Split
from clddp.evaluation import RetrievalMetric


@dataclass
class BM25DocPassageFusionArguments(AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn):
    data_dir: Optional[str] = None
    passage_results: Optional[str] = None
    split: Split = Split.test
    topk: int = 1000
    per_device_eval_batch_size: int = 32
    fp16: bool = True
    report_metric: str = RetrievalMetric.ndcg_string.at(10)
    report_passage_weight: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()

    @property
    def escaped_args(self) -> Set[str]:
        return {"passage_results"}
