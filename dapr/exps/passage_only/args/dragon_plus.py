from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.passage_only.args.base import PassageOnlyArguments


@dataclass
class DRAGONPlusPassageOnlyArguments(PassageOnlyArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/passage_only/dragon_plus", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(DRAGONPlusPassageOnlyArguments).output_dir
    )  # For creating the logging path
