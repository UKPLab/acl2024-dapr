from dataclasses import dataclass
import os
from typing import List
from clddp.utils import parse_cli
from dapr.exps.doc_retrieval_with_titles.args.base import (
    DocRetrievalWithTitlesArguments,
)


@dataclass
class ColBERTv2DocRetrievalWithTitlesArguments(DocRetrievalWithTitlesArguments):
    query_max_length: int = 150
    passage_max_length: int = 512

    def build_output_dir(self) -> str:
        return os.path.join("exps/doc_retrieval_with_titles/colbertv2", self.run_name)

    def get_arguments_from(self) -> List[type]:
        return [
            DocRetrievalWithTitlesArguments,
            ColBERTv2DocRetrievalWithTitlesArguments,
        ]


if __name__ == "__main__":
    print(
        parse_cli(ColBERTv2DocRetrievalWithTitlesArguments).output_dir
    )  # For creating the logging path
