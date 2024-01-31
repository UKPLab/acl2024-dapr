from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.coref.args.base import CorefArguments


@dataclass
class ColBERTv2CorefArguments(CorefArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/coref/colbertv2", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(ColBERTv2CorefArguments).output_dir
    )  # For creating the logging path
