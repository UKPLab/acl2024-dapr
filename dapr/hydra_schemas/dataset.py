from dataclasses import dataclass
import inspect
from typing import Optional, Type
from dapr.datasets.coliee import COLIEE
from dapr.datasets.miracl import MIRACL
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from dapr.datasets.base import BaseDataset
from dapr.datasets.nq import NaturalQuestions
from dapr.datasets.msmarco import MSMARCO
from dapr.datasets.genomics import Genomics
from dapr.utils import Separator, build_init_args_with_kwargs_and_default


@dataclass
class DatasetConfig:
    resource_path: str = MISSING
    min_plabel: int = MISSING  # Minimum of positive label
    nheldout: Optional[int] = None
    cache_root_dir: str = "data"
    chunk_size: int = 384
    tokenizer: str = "roberta-base"
    chunk_separator: Separator = Separator.empty
    nprocs: int = 12

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        raise NotImplementedError

    def __call__(self) -> BaseDataset:
        kwargs = build_init_args_with_kwargs_and_default(
            self.dataset_class.__init__,
            kwargs_specified=self.__dict__,
            kwargs_default=self.__dict__,
        )
        dataset = self.dataset_class(**kwargs)
        return dataset


@dataclass
class NaturalQuestionsConfig(DatasetConfig):
    resource_path: str = (
        "https://huggingface.co/datasets/sentence-transformers/NQ-retrieval"
    )
    # resource_path: str = "/home/fb20user07/research/nils-nq/NQ-retrieval"
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return NaturalQuestions


@dataclass
class MSMARCOConfig(DatasetConfig):
    resource_path: str = "https://msmarco.blob.core.windows.net"
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return MSMARCO


@dataclass
class GenomicsConfig(DatasetConfig):
    resource_path: str = ""  # We rely on ir_datasets
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return Genomics


@dataclass
class COLIEEConfig(DatasetConfig):
    resource_path: str = ""  # COLIEE store files in separate addresses
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return COLIEE


@dataclass
class MIRACLConfig(DatasetConfig):
    resource_path: str = "https://huggingface.co/datasets/miracl"
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return MIRACL


def register_dataset_config():
    cs = ConfigStore.instance()
    cs.store(group="dataset", name="nq", node=NaturalQuestionsConfig)
    cs.store(group="dataset", name="msmarco", node=MSMARCOConfig)
    cs.store(group="dataset", name="genomics", node=GenomicsConfig)
    cs.store(group="dataset", name="coliee", node=COLIEEConfig)
    cs.store(group="dataset", name="miracl", node=MIRACLConfig)
