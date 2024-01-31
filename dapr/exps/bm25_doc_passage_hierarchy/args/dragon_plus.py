from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.bm25_doc_passage_hierarchy.args.base import (
    BM25DocPassageHierarchyArguments,
)


@dataclass
class DRAGONPlusBM25DocPassageHierarchyArguments(BM25DocPassageHierarchyArguments):
    def build_output_dir(self) -> str:
        return os.path.join(
            "exps/bm25_doc_passage_hierarchy/dragon_plus", self.run_name
        )


if __name__ == "__main__":
    print(
        parse_cli(DRAGONPlusBM25DocPassageHierarchyArguments).output_dir
    )  # For creating the logging path
