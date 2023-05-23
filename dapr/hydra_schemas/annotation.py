from dataclasses import dataclass
from typing import Tuple
from dapr.annotators.base import BaseAnnotator
from dapr.annotators.empty import EmptyAnnotator
from dapr.annotators.lead import LeadingSentencesAnnotator
from dapr.annotators.pke import KeyphraseApproach, PKEAnnotator
from dapr.annotators.doc2query import Doc2QueryAnnotator
from dapr.annotators.title import TitleAnnotator
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class AnnotatorConfig:
    name: str = MISSING
    nprocs: int = 32

    def __call__(self) -> BaseAnnotator:
        raise NotImplementedError


@dataclass
class TopicRankAnnotatorConfig(AnnotatorConfig):
    name: str = "topic_rank"
    top_k_words: int = 10
    keyphrase_approach: KeyphraseApproach = KeyphraseApproach.topic_rank

    def __call__(self) -> PKEAnnotator:
        return PKEAnnotator(self.top_k_words, self.keyphrase_approach, self.nprocs)


@dataclass
class TitleAnnotatorConfig(AnnotatorConfig):
    name: str = "title"

    def __call__(self) -> TitleAnnotator:
        return TitleAnnotator()


@dataclass
class EmptyAnnotatorConfig(AnnotatorConfig):
    name: str = "empty"

    def __call__(self) -> EmptyAnnotator:
        return EmptyAnnotator()


@dataclass
class LeadingSentencesAnnotatorConfig(AnnotatorConfig):
    name: str = "lead"
    nlead: int = 3

    def __call__(self) -> LeadingSentencesAnnotator:
        return LeadingSentencesAnnotator(self.nlead)


@dataclass
class Doc2QueryAnnotatorConfig(AnnotatorConfig):
    name: str = "doc2query"
    nsamples: int = 40
    keep_ratio: float = 0.3
    batch_size_chunk = 16

    def __call__(self) -> Doc2QueryAnnotator:
        return Doc2QueryAnnotator(
            nsamples=self.nsamples,
            keep_ratio=self.keep_ratio,
            batch_size_chunk=self.batch_size_chunk,
        )


def register_annotator_config():
    cs = ConfigStore.instance()
    cs.store(
        group="annotator",
        name=TopicRankAnnotatorConfig.name,
        node=TopicRankAnnotatorConfig,
    )
    cs.store(
        group="annotator", name=TitleAnnotatorConfig.name, node=TitleAnnotatorConfig
    )
    cs.store(
        group="annotator", name=EmptyAnnotatorConfig.name, node=EmptyAnnotatorConfig
    )
    cs.store(
        group="annotator",
        name=LeadingSentencesAnnotatorConfig.name,
        node=LeadingSentencesAnnotatorConfig,
    )
    cs.store(
        group="annotator",
        name=Doc2QueryAnnotatorConfig.name,
        node=Doc2QueryAnnotatorConfig,
    )
