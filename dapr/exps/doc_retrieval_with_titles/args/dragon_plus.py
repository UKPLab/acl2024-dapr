from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.doc_retrieval_with_titles.args.base import (
    DocRetrievalWithTitlesArguments,
)


@dataclass
class DRAGONPlusDocRetrievalWithTitlesArguments(DocRetrievalWithTitlesArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/doc_retrieval_with_titles/dragon_plus", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(DRAGONPlusDocRetrievalWithTitlesArguments).output_dir
    )  # For creating the logging path
