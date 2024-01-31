from dataclasses import dataclass
import os
from typing import List
from clddp.utils import parse_cli
from dapr.exps.passage_only.args.base import PassageOnlyArguments


@dataclass
class JinaV2PassageOnlyArguments(PassageOnlyArguments):
    max_length: int = 512

    def get_arguments_from(self) -> List[type]:
        return [PassageOnlyArguments, JinaV2PassageOnlyArguments]

    def build_output_dir(self) -> str:
        return os.path.join("exps/passage_only/jinav2", self.run_name)


if __name__ == "__main__":
    print(
        parse_cli(JinaV2PassageOnlyArguments).output_dir
    )  # For creating the logging path
