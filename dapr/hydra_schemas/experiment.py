from dataclasses import dataclass
from enum import Enum
from hydra.core.config_store import ConfigStore
from dapr.datasets.dm import Split
from dapr.models.evaluation import RetrievalMetric
from typing import Optional, Tuple


class Seed(str, Enum):
    seed1 = "seed1"
    seed2 = "seed2"
    seed3 = "seed3"
    seed4 = "seed4"
    seed5 = "seed5"
    seed6 = "seed6"

    def __call__(self) -> int:
        # import random; rs = random.Random(42); seeds = [rs.randint(0, 1000) for _ in range(len(Seed))]
        seeds = [654, 114, 25, 759, 281, 250]
        return dict(zip(list(Seed), seeds))[self]


@dataclass
class ExperimentConfig:
    seed: Seed = Seed.seed1
    project_name: str = "dapr"  # Original paragraphs (except MSMARCO)
    batch_size: int = 75
    mini_batch_size: int = 6
    epochs: int = 10
    warmup: int = 1000

    split: Split = Split.test
    metrics: Tuple[str] = (
        RetrievalMetric.ndcg_string.at(10),
        RetrievalMetric.rr_string.at(10),
        RetrievalMetric.recall_string.at(100),
    )
    main_metric: str = RetrievalMetric.ndcg_string.at(10)
    wandb: bool = False

    ckpt_dir: Optional[str] = None
    use_amp: bool = True

    results_dir: str = "results"
    topk: int = 1000


def register_experiment_config():
    cs = ConfigStore.instance()
    cs.store(name="experiment", node=ExperimentConfig)
