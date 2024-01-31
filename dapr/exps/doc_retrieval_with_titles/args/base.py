from dataclasses import dataclass
from typing import Optional
from clddp.args.base import AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn
from clddp.dm import Split


@dataclass
class DocRetrievalWithTitlesArguments(
    AutoRunNameArgumentsMixIn, DumpableArgumentsMixIn
):
    data_dir: Optional[str] = None
    split: Split = Split.test
    topk: int = 1000
    per_device_eval_batch_size: int = 32
    fp16: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
