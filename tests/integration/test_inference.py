import os
import shutil
import sys
from typing import Optional, Type
from unittest import mock
from dapr.hydra_schemas.dataset import NaturalQuestionsConfig
from dapr.hydra_schemas.retrieval import ColBERTV2Config
from dapr.models.evaluation import RetrievalMetric
from dapr.models.retrieval.base import DocumentMethod
import pytest
from tests.conftest import DatasetFixture
from tests.integration.test_datasets import build_nq
from dapr.inference.base import BaseExperiment
from dapr.inference.bm25_chunk_neural_chunk_max_p import BM25ChunkNeuralChunkMaxP
from dapr.inference.bm25_neural_first_p import BM25NeuralFirstP
from dapr.inference.bm25_doc_neural_max_p import BM25DocNeuralMaxP
from dapr.inference.bm25_doc_neural_chunk_max_p import BM25DocNeuralChunkMaxP
from dapr.inference.bm25 import BM25
from dapr.inference.bm25_doc_bm25_chunk import BM25DocBM25Chunk
from dapr.inference.bm25_summarized_context import BM25SummarizedContext
from dapr.inference.summarized_context_max_p import SummarizedContextMaxP
from dapr.inference.sparse_pooling import SparsePooling
from dapr.hydra_schemas.annotation import (
    EmptyAnnotatorConfig,
    TitleAnnotatorConfig,
    LeadingSentencesAnnotatorConfig,
    TopicRankAnnotatorConfig,
)

from hydra.core.global_hydra import GlobalHydra
import torch


@pytest.mark.parametrize(
    "experiment_class, annotator, retriever, doc_method",
    [
        (
            BM25DocNeuralMaxP,
            EmptyAnnotatorConfig.name,
            "colbertv2",
            None,
        ),
        (
            SparsePooling,
            EmptyAnnotatorConfig.name,
            "splade-cocondenser-ensembledistil",
            DocumentMethod.pooling_max,
        ),
        (
            SparsePooling,
            EmptyAnnotatorConfig.name,
            "splade-cocondenser-ensembledistil",
            DocumentMethod.pooling_mean,
        ),
        (
            SparsePooling,
            EmptyAnnotatorConfig.name,
            "splade-cocondenser-ensembledistil",
            DocumentMethod.pooling_sum,
        ),
        (
            BM25DocNeuralMaxP,
            EmptyAnnotatorConfig.name,
            "splade-cocondenser-ensembledistil",
            None,
        ),
        (BM25, EmptyAnnotatorConfig.name, "nq-distilbert-base-v1", None),
        (BM25DocBM25Chunk, EmptyAnnotatorConfig.name, "nq-distilbert-base-v1", None),
        (
            BM25ChunkNeuralChunkMaxP,
            EmptyAnnotatorConfig.name,
            "nq-distilbert-base-v1",
            None,
        ),
        (BM25NeuralFirstP, EmptyAnnotatorConfig.name, "nq-distilbert-base-v1", None),
        (BM25DocNeuralMaxP, EmptyAnnotatorConfig.name, "nq-distilbert-base-v1", None),
        (
            BM25DocNeuralChunkMaxP,
            EmptyAnnotatorConfig.name,
            "nq-distilbert-base-v1",
            None,
        ),
        (
            SummarizedContextMaxP,
            EmptyAnnotatorConfig.name,
            "nq-distilbert-base-v1",
            None,
        ),
        (
            SummarizedContextMaxP,
            TitleAnnotatorConfig.name,
            "nq-distilbert-base-v1",
            None,
        ),
        (
            SummarizedContextMaxP,
            LeadingSentencesAnnotatorConfig.name,
            "nq-distilbert-base-v1",
            None,
        ),
        (
            SummarizedContextMaxP,
            TopicRankAnnotatorConfig.name,
            "nq-distilbert-base-v1",
            None,
        ),
        (
            BM25SummarizedContext,
            LeadingSentencesAnnotatorConfig.name,
            "nq-distilbert-base-v1",
            None,
        ),
    ],
)
def test_inference_approaches(
    nq_resource: DatasetFixture,
    cache_root_dir: str,
    results_dir: str,
    distilbert_path: str,
    experiment_class: Type[BaseExperiment],
    annotator: str,
    retriever: str,
    doc_method: Optional[str],
):
    assert torch.cuda.is_available()
    dataset = build_nq(data_path=nq_resource.data_path, cache_root_dir=cache_root_dir)
    mock_device_count = mock.patch.object(torch.cuda, "device_count", return_value=1)
    sys_argv = [
        "",
        "+dataset=nq",
        f"+retriever={retriever}",
        f"+annotator={annotator}",
        "experiment.wandb=False",
        f"experiment.main_metric={RetrievalMetric.ndcg_string.at(1)}",
        f"experiment.metrics={[RetrievalMetric.ndcg_string.at(1), RetrievalMetric.rr_string.at(2)]}",
        f"experiment.results_dir={results_dir}",
        "experiment.topk=2",
    ]
    if retriever != ColBERTV2Config.name:
        sys_argv.append(f"retriever.query_model={distilbert_path}")
    if doc_method:
        sys_argv.append(f"retriever.doc_method={doc_method}")
    mock_sys_argv = mock.patch.object(sys, "argv", sys_argv)
    mock_nq_instantiation = mock.patch.object(
        NaturalQuestionsConfig, "__call__", return_value=dataset
    )

    with mock_sys_argv, mock_nq_instantiation, mock_device_count:
        exp = experiment_class()
        try:
            exp.run()
        finally:
            if exp.bm25 is not None:
                exp.bm25.clear_cache()
            GlobalHydra.instance().clear()
            if os.path.exists(cache_root_dir):
                shutil.rmtree(cache_root_dir)
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
