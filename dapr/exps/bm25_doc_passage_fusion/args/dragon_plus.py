from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.bm25_doc_passage_fusion.args.base import (
    BM25DocPassageFusionArguments,
)


@dataclass
class DRAGONPlusBM25DocPassageFusionArguments(BM25DocPassageFusionArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/bm25_doc_passage_fusion/dragon_plus", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(DRAGONPlusBM25DocPassageFusionArguments).output_dir
    )  # For creating the logging path
