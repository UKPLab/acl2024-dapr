from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.passage_only.args.base import PassageOnlyArguments


@dataclass
class BM25PassageOnlyArguments(PassageOnlyArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/passage_only/bm25", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(BM25PassageOnlyArguments).output_dir
    )  # For creating the logging path
