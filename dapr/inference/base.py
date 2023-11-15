import logging
import os
import sys
from typing import Any, Dict, List, Optional

from attr import dataclass
from dapr.annotators.base import BaseAnnotator
from dapr.datasets.base import BaseDataset
from dapr.hydra_schemas.annotation import AnnotatorConfig, register_annotator_config

from dapr.hydra_schemas.experiment import (
    register_experiment_config,
    ExperimentConfig,
)
from dapr.hydra_schemas.retrieval import BM25Config, register_retriever_config
from dapr.hydra_schemas.dataset import register_dataset_config
from dapr.hydra_schemas.retrieval import RetrieverConfig
from dapr.hydra_schemas.dataset import DatasetConfig
from dapr.models.dm import RetrievalLevel
from dapr.models.retrieval.bm25 import BM25Retriever
from dapr.models.retrieval.neural import NeuralRetriever
from hydra import compose, initialize

import wandb
from dapr.datasets.dm import Split
from dapr.models.evaluation import EvaluationOutput, LongDocumentEvaluator
from dapr.utils import set_logger_format, switch, wandb_report
from transformers import set_seed
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from abc import ABC, abstractmethod


@dataclass
class Config:
    dataset: DatasetConfig = MISSING
    retriever: RetrieverConfig = MISSING  # Neural retriever
    annotator: AnnotatorConfig = MISSING
    experiment: ExperimentConfig = ExperimentConfig()


def register_config() -> None:
    register_dataset_config()
    register_retriever_config()
    register_annotator_config()
    register_experiment_config()
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)


class BaseExperiment(ABC):
    bm25: Optional[BM25Retriever] = None
    neural_retriever: Optional[NeuralRetriever] = None
    cfg: Optional[Config] = None

    def build_run_name(self, cfg: Config) -> str:
        choices: Dict[str, Optional[str]] = getattr(
            getattr(getattr(cfg, "hydra"), "runtime"), "choices"
        )
        script_name = self.__class__.__name__
        items = []
        if "retriever" in choices:
            items.append(choices["retriever"])
        items.append(choices["dataset"])
        items.append(script_name)
        if "annotator" in choices:
            items.append(choices["annotator"])
        if "retriever" in choices:
            # TODO: Merge this with the previous if-retriever-clause
            if cfg.retriever.doc_method:
                items.append(cfg.retriever.doc_method)
        items.append(cfg.experiment.seed)
        items.append(cfg.experiment.split)
        run_name = "-".join(items)
        return run_name

    def run(self) -> List[EvaluationOutput]:
        set_logger_format()
        self.logger = logging.getLogger(__name__)

        # Build Hydra config:
        register_config()
        initialize(version_base=None)
        cfg: Config = compose(
            config_name="config", overrides=sys.argv[1:], return_hydra_config=True
        )
        self.cfg = cfg

        # Initialize wandb:
        seed = cfg.experiment.seed()
        set_seed(seed)
        all_config = OmegaConf.to_container(cfg)
        all_config["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        run_name = self.build_run_name(cfg)
        switch(cfg.experiment.wandb)(wandb.init)(
            project=cfg.experiment.project_name,
            name=run_name,
            config=all_config,
        )

        # Score and report:
        eouts = self.score(cfg=cfg)
        report = {}
        for eout in eouts:
            report.update(eout.summary)
        switch(cfg.experiment.wandb)(wandb_report)(report)
        return eouts

    def build_bm25_retriever(self) -> BM25Retriever:
        self.bm25 = BM25Config()()
        return self.bm25

    def build_neural_retriever(self, cfg: Config, **kwargs: Any) -> NeuralRetriever:
        retriever_class: RetrieverConfig = OmegaConf.to_object(cfg.retriever)
        self.neural_retriever = retriever_class(**kwargs)
        return self.neural_retriever

    def build_annotator(self, cfg: Config) -> BaseAnnotator:
        annotator = OmegaConf.to_object(cfg.annotator)()
        return annotator

    def build_evaluator(
        self,
        cfg: Config,
        annotator: Optional[BaseAnnotator] = None,
    ) -> LongDocumentEvaluator:
        self.logger.info("Building dataset")
        dataset: BaseDataset = OmegaConf.to_object(cfg.dataset)()

        if annotator is not None:
            self.logger.info("Annotating")
            annotator.annotate(
                data=dataset.loaded_data, cache_root_dir=cfg.experiment.results_dir
            )

        self.logger.info("Instantiating evaluator")
        evaluator = LongDocumentEvaluator(
            data=dataset.loaded_data,
            results_dir=cfg.experiment.results_dir,
            split=cfg.experiment.split,
            min_plabel=cfg.dataset.min_plabel,
            metrics=cfg.experiment.metrics,
            main_metric=cfg.experiment.main_metric,
            topk=cfg.experiment.topk,
        )
        return evaluator

    def build_retrieval_kwargs(
        self, evaluator: LongDocumentEvaluator
    ) -> Dict[str, Any]:
        kwargs = {
            "queries": evaluator.queries,
            "pool": evaluator.pool,
            "ndocs": evaluator.ndocs,
            "nchunks": evaluator.nchunks,
            "pool_identifier": evaluator.pool_identifier,
            "chunk_separator": evaluator.data.meta_data["chunk_separator"],
            "results_dir": evaluator.results_dir,
            "topk": evaluator.topk,
        }
        return kwargs

    @abstractmethod
    def score(self, cfg: Config) -> List[EvaluationOutput]:
        pass
