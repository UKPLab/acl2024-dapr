from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.prepending_titles.args.base import PrependingTitlesArguments


@dataclass
class SPLADEv2lusPrependingTitlesArguments(PrependingTitlesArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/prepending_titles/spladev2", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(SPLADEv2lusPrependingTitlesArguments).output_dir
    )  # For creating the logging path
