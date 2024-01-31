from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.keyphrases.args.base import KeyphrasesArguments


@dataclass
class ColBERTv2KeyphrasesArguments(KeyphrasesArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/keyphrases/colbertv2", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(ColBERTv2KeyphrasesArguments).output_dir
    )  # For creating the logging path
