from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.bm25_doc_passage_fusion.args.base import (
    BM25DocPassageFusionArguments,
)


@dataclass
class CoLBERTv2BM25DocPassageFusionArguments(BM25DocPassageFusionArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/bm25_doc_passage_fusion/colbertv2", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(CoLBERTv2BM25DocPassageFusionArguments).output_dir
    )  # For creating the logging path
