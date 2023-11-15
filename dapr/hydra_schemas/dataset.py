from dataclasses import dataclass
import inspect
from typing import Optional, Type
from dapr.datasets.conditionalqa import ConditionalQA
from dapr.datasets.cleaned_conditionalqa import CleanedConditionalQA
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
    name: str = MISSING
    resource_path: str = MISSING
    min_plabel: int = MISSING  # Minimum of positive label
    nheldout: Optional[int] = None
    cache_root_dir: str = "data"
    chunk_size: int = 384
    tokenizer: str = "roberta-base"
    chunk_separator: Separator = Separator.empty
    nprocs: int = 10

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
    name: str = "nq"
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
    name: str = "msmarco"
    resource_path: str = "https://msmarco.blob.core.windows.net"
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return MSMARCO


@dataclass
class GenomicsConfig(DatasetConfig):
    name: str = "genomics"
    resource_path: str = ""  # We rely on ir_datasets
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return Genomics


@dataclass
class MIRACLConfig(DatasetConfig):
    name: str = "miracl"
    resource_path: str = "https://huggingface.co/datasets/miracl"
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return MIRACL


@dataclass
class ConditionalQAConfig(DatasetConfig):
    name: str = "conditionalqa"
    resource_path: str = (
        "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0"
    )
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return ConditionalQA


@dataclass
class CleanedConditionalQAConfig(DatasetConfig):
    name: str = "cleaned_conditionalqa"
    resource_path: str = (
        "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0"
    )
    min_plabel: int = 1

    @property
    def dataset_class(self) -> Type[BaseDataset]:
        return CleanedConditionalQA


def register_dataset_config(group: str = "dataset"):
    cs = ConfigStore.instance()
    cs.store(group=group, name=NaturalQuestionsConfig.name, node=NaturalQuestionsConfig)
    cs.store(group=group, name=MSMARCOConfig.name, node=MSMARCOConfig)
    cs.store(group=group, name=GenomicsConfig.name, node=GenomicsConfig)
    cs.store(group=group, name=MIRACLConfig.name, node=MIRACLConfig)
    cs.store(group=group, name=ConditionalQAConfig.name, node=ConditionalQAConfig)
    cs.store(
        group=group,
        name=CleanedConditionalQAConfig.name,
        node=CleanedConditionalQAConfig,
    )
