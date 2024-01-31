from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.keyphrases.args.base import KeyphrasesArguments


@dataclass
class DRAGONPlusKeyphrasesArguments(KeyphrasesArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/keyphrases/dragon_plus", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(DRAGONPlusKeyphrasesArguments).output_dir
    )  # For creating the logging path
