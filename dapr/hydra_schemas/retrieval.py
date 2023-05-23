from dataclasses import MISSING, dataclass
from typing import Any, Dict, Optional

# from ldr.models.retrieval.neural import NeuralRetriever
from dapr.models.retrieval.base import BaseRetriever, DocumentMethod
from dapr.models.retrieval.bm25 import BM25Retriever
from dapr.models.retrieval.neural import NeuralRetriever
from dapr.utils import Separator, build_init_args_with_kwargs_and_default
from dapr.models.encoding import (
    SimilarityFunction,
    Pooling,
    EncoderType,
    TextEncoder,
)
from hydra.core.config_store import ConfigStore
import inspect


@dataclass
class RetrieverConfig:
    name: str = MISSING
    mnrl_scale: float = MISSING
    query_model: Optional[str] = MISSING
    document_model: Optional[str] = MISSING
    similarity_function: SimilarityFunction = MISSING
    first_ntokens_global: Optional[int] = MISSING
    pooling: Pooling = MISSING

    encoder_type: Optional[EncoderType] = MISSING  # None means pyserini
    chunk_separator: Separator = Separator.empty
    title_body_separator: Separator = Separator.blank
    max_length: Optional[int] = 512
    max_nchunks: Optional[int] = None
    lambda_d: float = 0
    lambda_q: float = 0

    batch_size_query: int = 64
    batch_size_chunk: int = 64
    doc_method: Optional[DocumentMethod] = None
    bm25_weight: float = 1.0

    def build_encoder(
        self,
        encoder_type: EncoderType,
        kwargs_specified: Dict[str, Any],
        kwargs_default: Dict[str, Any],
        text_type: str,
    ) -> Optional[TextEncoder]:
        assert text_type in ["query", "document"]
        model_arg = f"{text_type}_model"
        encoder_class = encoder_type()
        kwargs_specified = dict(kwargs_specified)
        kwargs_default = dict(kwargs_default)
        kwargs_specified["model_name_or_path"] = kwargs_specified.pop(model_arg)
        kwargs_default["model_name_or_path"] = kwargs_default.pop(model_arg)
        kwargs_encoder = build_init_args_with_kwargs_and_default(
            init_fn=encoder_class.__init__,
            kwargs_specified=kwargs_specified,
            kwargs_default=kwargs_default,
        )
        if kwargs_encoder["model_name_or_path"] is None:
            return None
        encoder = encoder_class(**kwargs_encoder)
        return encoder

    def __call__(
        self,
        name: Optional[str] = None,
        mnrl_scale: Optional[float] = None,
        query_model: Optional[str] = None,
        document_model: Optional[str] = None,
        similarity_function: Optional[SimilarityFunction] = None,
        first_ntokens_global: Optional[int] = None,
        pooling: Optional[Pooling] = None,
        chunk_separator: Optional[Separator] = None,
        title_body_separator: Optional[Separator] = None,
        max_length: Optional[int] = None,
        max_nchunks: Optional[int] = None,
        lambda_d: Optional[float] = None,
        lambda_q: Optional[float] = None,
        batch_size_query: Optional[int] = None,
        batch_size_chunk: Optional[int] = None,
        doc_method: Optional[DocumentMethod] = None,
        bm25_weight: Optional[float] = None,
    ) -> BaseRetriever:
        kwargs_specified = dict(inspect.getargvalues(inspect.currentframe()).locals)
        kwargs_default = self.__dict__

        query_encoder = self.build_encoder(
            encoder_type=self.encoder_type,
            kwargs_specified=kwargs_specified,
            kwargs_default=kwargs_default,
            text_type="query",
        )
        document_encoder = self.build_encoder(
            encoder_type=self.encoder_type,
            kwargs_specified=kwargs_specified,
            kwargs_default=kwargs_default,
            text_type="document",
        )
        kwargs_neural = build_init_args_with_kwargs_and_default(
            init_fn=NeuralRetriever.__init__,
            kwargs_specified=kwargs_specified,
            kwargs_default=kwargs_default,
        )
        kwargs_neural["query_encoder"] = query_encoder
        kwargs_neural["document_encoder"] = document_encoder
        return NeuralRetriever(**kwargs_neural)


@dataclass
class BM25Config(RetrieverConfig):
    name: str = "bm25"
    encoder_type: Optional[EncoderType] = None
    mnrl_scale: float = -1
    query_model: Optional[str] = None
    document_model: Optional[str] = None
    similarity_function: SimilarityFunction = (
        SimilarityFunction.dot_product
    )  # dummy argument
    first_ntokens_global: Optional[int] = None  # dummy argument
    pooling: Pooling = Pooling.mean  # dummy argument

    bm25_weight: float = 1.0

    def __call__(
        self,
        name: Optional[str] = None,
        mnrl_scale: Optional[float] = None,
        query_model: Optional[str] = None,
        document_model: Optional[str] = None,
        similarity_function: Optional[SimilarityFunction] = None,
        first_ntokens_global: Optional[int] = None,
        pooling: Optional[Pooling] = None,
        chunk_separator: Optional[Separator] = None,
        title_body_separator: Optional[Separator] = None,
        max_length: Optional[int] = None,
        max_nchunks: Optional[int] = None,
        lambda_d: Optional[float] = None,
        lambda_q: Optional[float] = None,
        batch_size_query: Optional[int] = None,
        batch_size_chunk: Optional[int] = None,
        doc_method: Optional[DocumentMethod] = None,
        bm25_weight: Optional[float] = None,
    ) -> BaseRetriever:
        kwargs_specified = dict(inspect.getargvalues(inspect.currentframe()).locals)
        kwargs_default = self.__dict__

        kwargs_bm25 = build_init_args_with_kwargs_and_default(
            init_fn=BM25Retriever.__init__,
            kwargs_specified=kwargs_specified,
            kwargs_default=kwargs_default,
        )
        assert self.encoder_type is None
        return BM25Retriever(**kwargs_bm25)


@dataclass
class NQDistilBERTConfig(RetrieverConfig):
    name: str = "nq-distilbert-base-v1"
    encoder_type: Optional[EncoderType] = EncoderType.single_vector
    mnrl_scale: float = 20
    query_model: str = "sentence-transformers/nq-distilbert-base-v1"
    document_model: Optional[str] = None
    similarity_function: SimilarityFunction = SimilarityFunction.cos_sim
    first_ntokens_global: Optional[int] = None
    pooling: Pooling = Pooling.mean
    title_body_separator: Separator = Separator.bert_sep
    max_length: Optional[int] = 512


@dataclass
class RetroMAEBEIRConfig(RetrieverConfig):
    name: str = "retromae-beir"
    encoder_type: Optional[EncoderType] = EncoderType.single_vector
    mnrl_scale: float = 1
    query_model: str = "Shitao/RetroMAE_BEIR"
    document_model: Optional[str] = None
    similarity_function: SimilarityFunction = SimilarityFunction.dot_product
    first_ntokens_global: Optional[int] = None
    pooling: Pooling = Pooling.cls
    title_body_separator: Separator = Separator.bert_sep
    max_length: Optional[int] = 512


@dataclass
class SPLADECoConDenserEnsembleDistilConfig(RetrieverConfig):
    name: str = "splade-cocondenser-ensembledistil"
    encoder_type: Optional[EncoderType] = EncoderType.single_vector
    mnrl_scale: float = 1
    query_model: str = "naver/splade-cocondenser-ensembledistil"
    document_model: Optional[str] = None
    similarity_function: SimilarityFunction = SimilarityFunction.dot_product
    first_ntokens_global: Optional[int] = None
    pooling: Pooling = Pooling.splade
    title_body_separator: Separator = Separator.blank
    # max_length: Optional[int] = 300
    max_length: Optional[int] = 512


@dataclass
class ColBERTV2Config(RetrieverConfig):
    name: str = "colbertv2"
    encoder_type: Optional[EncoderType] = EncoderType.colbert
    mnrl_scale: float = 1
    query_model: str = "https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz"
    document_model: Optional[str] = None
    similarity_function: SimilarityFunction = SimilarityFunction.maxsim
    first_ntokens_global: Optional[int] = None
    pooling: Pooling = Pooling.no_pooling  # dummy
    title_body_separator: Separator = Separator.blank
    max_length: Optional[int] = 512


def register_retriever_config():
    cs = ConfigStore.instance()
    cs.store(group="retriever", name=BM25Config.name, node=BM25Config)
    cs.store(group="retriever", name=NQDistilBERTConfig.name, node=NQDistilBERTConfig)
    cs.store(group="retriever", name=RetroMAEBEIRConfig.name, node=RetroMAEBEIRConfig)
    cs.store(
        group="retriever",
        name=SPLADECoConDenserEnsembleDistilConfig.name,
        node=SPLADECoConDenserEnsembleDistilConfig,
    )
    cs.store(group="retriever", name=ColBERTV2Config.name, node=ColBERTV2Config)
