from dataclasses import dataclass
import os
from clddp.utils import parse_cli
from dapr.exps.passage_only.args.base import PassageOnlyArguments


@dataclass
class SPLADEv2PassageOnlyArguments(PassageOnlyArguments):
    def build_output_dir(self) -> str:
        return os.path.join("exps/passage_only/spladev2", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(SPLADEv2PassageOnlyArguments).output_dir
    )  # For creating the logging path
