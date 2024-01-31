from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.prepending_titles.args.base import PrependingTitlesArguments


@dataclass
class DRAGONPlusPrependingTitlesArguments(PrependingTitlesArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/prepending_titles/dragon_plus", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(DRAGONPlusPrependingTitlesArguments).output_dir
    )  # For creating the logging path
