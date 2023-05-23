from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import chain
import json
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional, Type, Union, TypedDict
from unittest import mock
from colbert.modeling import base_colbert
from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score
from colbert.search.strided_tensor import StridedTensor
from dapr.utils import ColBERTQueryTokenizer, Pooling, Separator, download, HF_ColBERT
from datasets import DownloadManager
import torch
import tqdm
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    LongformerModel,
    LongformerConfig,
    BatchEncoding,
)
from dapr.datasets.dm import Chunk, Document, Query
from transformers.models.distilbert.modeling_distilbert import DistilBertConfig
from transformers.models.xlnet.configuration_xlnet import XLNetConfig
from transformers.models.longformer.modeling_longformer import (
    LongformerBaseModelOutputWithPooling,
    LongformerConfig,
    LongformerMaskedLMOutput,
)
from sentence_transformers.util import (
    cos_sim,
    dot_score,
    pairwise_cos_sim,
    pairwise_dot_score,
)
import math


@dataclass
class TextFeatures:
    input_ids: torch.Tensor  # (bsz, chunk_size)
    attention_mask: torch.Tensor  # (bsz, chunk_size)
    other_cls_positions: Optional[torch.Tensor] = None  # (bsz, nchunks)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, subscript: Union[slice, int]) -> TextFeatures:
        return TextFeatures(
            input_ids=self.input_ids[subscript],
            attention_mask=self.attention_mask[subscript],
            other_cls_positions=self.other_cls_positions[subscript]
            if self.other_cls_positions is not None
            else None,
        )


def pairwise_maxsim(
    qembs: torch.Tensor, cembs: torch.Tensor, cmask: torch.BoolTensor
) -> torch.Tensor:
    assert cmask is not None
    return colbert_score(Q=qembs, D_padded=cembs, D_mask=cmask)


def maxsim(
    qembs: torch.Tensor,
    cembs: torch.Tensor,
    cmask: torch.BoolTensor,
    batch_size_query: int = 1024,
    batch_size_chunk: int = 16,
) -> torch.Tensor:
    assert cmask is not None
    scores = []
    for bq in range(0, len(qembs), batch_size_query):
        eq = bq + batch_size_query
        qbatch = qembs[bq:eq]
        scores_qbatch = []
        for bc in range(0, len(cembs), batch_size_chunk):
            ec = bc + batch_size_chunk
            scores_4d = cembs[None, bc:ec] @ qbatch[:, None].to(
                dtype=cembs.dtype
            ).transpose(
                2, 3
            )  # (nqueries, nchunks, chunk_length, query_length)
            scores_4d += (
                ~cmask[None, bc:ec] * -9999
            )  # (nchunks, chunk_length, 1) -> (1, nchunks, chunk_length, 1)
            scores_4d_max: torch.Tensor = scores_4d.max(dim=2)[
                0
            ]  # (nqueries, nchunks, query_length)
            scores_2d = scores_4d_max.sum(dim=-1)  # (nqueries, nchunks)
            scores_qbatch.append(scores_2d)
        scores.append(torch.cat(scores_qbatch, dim=-1))
    return torch.cat(scores, dim=0)


class SimilarityFunction(str, Enum):
    """Vector distance between embeddings."""

    dot_product = "dot_product"
    cos_sim = "cos_sim"
    maxsim = "maxsim"

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        chunk_embeddings: torch.Tensor,
        chunk_mask: Optional[torch.BoolTensor] = None,
        pairwise: bool = False,
    ) -> torch.Tensor:
        """Run the score function over query and passage embeddings."""
        if pairwise:
            # return shape: (npairs,)
            assert len(query_embeddings) == len(chunk_embeddings)
            fn = {
                SimilarityFunction.dot_product: pairwise_dot_score,
                SimilarityFunction.cos_sim: pairwise_cos_sim,
                SimilarityFunction.maxsim: partial(pairwise_maxsim, cmask=chunk_mask),
            }[self]
        else:
            # resturn shape: (nqueries, npassages)
            fn = {
                SimilarityFunction.dot_product: dot_score,
                SimilarityFunction.cos_sim: cos_sim,
                SimilarityFunction.maxsim: partial(maxsim, cmask=chunk_mask),
            }[self]
        scores = fn(query_embeddings, chunk_embeddings)
        return scores

    @property
    def __name__(self):
        return {
            SimilarityFunction.dot_product: dot_score,
            SimilarityFunction.cos_sim: cos_sim,
        }[self].__name__


@dataclass
class EmbeddingsWithMask:
    embeddings: torch.Tensor
    mask: Optional[torch.BoolTensor] = None


class TextEncoder(ABC, torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        similarity_function: SimilarityFunction,
        pooling: Pooling,
        max_length: int,
        max_nchunks: Optional[int],
        title_body_separator: Optional[Separator],
        chunk_separator: Optional[Separator],
    ) -> None:
        super().__init__()
        if model_name_or_path.endswith(".tar.gz") or model_name_or_path.endswith(
            ".zip"
        ):
            dm = DownloadManager()
            model_name_or_path = dm.download_and_extract(model_name_or_path)
            # it could be a folder after extraction:
            if len(os.listdir(model_name_or_path)) == 1:
                first_path = os.path.join(
                    model_name_or_path, os.listdir(model_name_or_path)[0]
                )
                if os.path.isdir(first_path):
                    model_name_or_path = first_path

        self.pooling = Pooling(pooling)
        self.similarity_function = SimilarityFunction(similarity_function)
        self.max_length = max_length
        self.model = self.load_model(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.title_body_separator = Separator(title_body_separator)
        self.chunk_separator = Separator(chunk_separator)
        self.max_nchunks = max_nchunks
        self.extra_config = {
            "title_body_separator": title_body_separator,
            "chunk_separator": chunk_separator,
            "pooling": pooling,
            "max_length": max_length,
            "similarity_function": similarity_function,
        }
        self.device: Optional[torch.device] = None

    @property
    def hidden_size(self) -> int:
        return getattr(self.model.config, "hidden_size")

    def set_device(self, device: Union[int, str]) -> None:
        self.device = torch.device(device)
        self.to(self.device)

    @abstractmethod
    def load_model(self, model_name_or_path: str) -> PreTrainedModel:
        pass

    @classmethod
    def load(cls: Type[TextEncoder], from_dir: str):
        with open(os.path.join(from_dir, "extra_config.json"), "r") as f:
            extra_config = json.load(f)
        return cls(model_name_or_path=from_dir, **extra_config)

    def tokenize_queries(self, queries: List[Query]) -> TextFeatures:
        query_texts = list(map(lambda query: query.text, queries))
        tokenized: BatchEncoding = self.tokenizer(
            query_texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)
        input_ids: torch.Tensor = tokenized["input_ids"]
        attention_mask: torch.Tensor = tokenized["attention_mask"]
        text_features = TextFeatures(input_ids=input_ids, attention_mask=attention_mask)
        return text_features

    def chunk2text(self, chunk: Chunk) -> str:
        return chunk.text

    def tokenize_documents(self, documents: List[Document]) -> TextFeatures:
        chunk_texts = []
        max_nchunks = max(len(doc.chunks) for doc in documents)
        if self.max_nchunks is not None:
            max_nchunks = min(self.max_nchunks, max_nchunks)

        # Build cls positions:
        other_cls_positions: List[List[int]] = []
        for doc in documents:
            cls_positions = [
                [
                    position + len(chunk_texts)
                    for position in range(max_nchunks)
                    if position != b
                ]
                for b in range(len(doc.chunks))
            ]
            other_cls_positions.extend(cls_positions)
            chunk_texts.extend(map(self.chunk2text, doc.chunks))

        # Build other features:
        tokenized: BatchEncoding = self.tokenizer(
            chunk_texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length,
        ).to(self.device)

        # Assembly:
        input_ids: torch.Tensor = tokenized["input_ids"]
        attention_mask: torch.Tensor = tokenized["attention_mask"]
        text_features = TextFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            other_cls_positions=torch.LongTensor(other_cls_positions).to(self.device),
        )
        return text_features

    @abstractmethod
    def encode_queries(
        self, queries: List[Query], batch_size_query: int, show_progress_bar=True
    ) -> torch.Tensor:
        """Encode queries into query embeddings."""
        pass

    @abstractmethod
    def encode_documents(
        self,
        documents: List[Document],
        batch_size_document: Optional[int] = None,
        batch_size_chunk: Optional[int] = None,
    ) -> EmbeddingsWithMask:
        """Encode documents into chunk embeddings."""
        pass


class SingleVectorEncoder(TextEncoder):
    def load_model(self, model_name_or_path: str) -> PreTrainedModel:
        if self.pooling is Pooling.splade:
            return AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        else:
            return AutoModel.from_pretrained(model_name_or_path)

    def chunk2text(self, chunk: Chunk) -> str:
        return self.title_body_separator.concat([chunk.doc_summary, chunk.text])

    def forward(self, text_features: TextFeatures) -> torch.Tensor:
        features = text_features.__dict__
        features.pop("other_cls_positions")
        token_embeddings: torch.Tensor = self.model(**features, return_dict=False)[0]
        pooled = self.pooling(
            token_embeddings=token_embeddings,
            attention_mask=text_features.attention_mask,
        )
        return pooled

    @torch.no_grad()
    def encode_queries(
        self, queries: List[Query], batch_size_query: int, show_progress_bar=True
    ) -> torch.Tensor:
        self.eval()
        qembs: Union[List[torch.Tensor], torch.Tensor] = []
        for b in tqdm.trange(
            0,
            len(queries),
            batch_size_query,
            desc="Encode queries",
            disable=not show_progress_bar,
        ):
            e = b + batch_size_query
            qbatch = queries[b:e]
            qbatch_embs = self.forward(self.tokenize_queries(qbatch))
            qembs.append(qbatch_embs)
        qembs = torch.cat(qembs, dim=0)
        return qembs

    @torch.no_grad()
    def encode_documents(
        self,
        documents: List[Document],
        batch_size_document: Optional[int] = None,
        batch_size_chunk: Optional[int] = None,
    ) -> EmbeddingsWithMask:
        """Encode documents into chunk embeddings."""
        assert batch_size_chunk is not None
        self.eval()
        chunk_embs = []
        features = self.tokenize_documents(documents)
        for b in range(0, len(features), batch_size_chunk):
            e = b + batch_size_chunk
            embs = self.forward(features[b:e])
            chunk_embs.append(embs)
        chunk_embs = torch.cat(chunk_embs, dim=0)
        return EmbeddingsWithMask(embeddings=chunk_embs)


class ColBERTEncoder(TextEncoder):
    def load_model(self, model_name_or_path: str) -> PreTrainedModel:
        assert self.similarity_function is SimilarityFunction.maxsim
        assert self.pooling is Pooling.no_pooling
        class_factory_mock = mock.patch.object(
            base_colbert, "class_factory", return_value=HF_ColBERT
        )
        with class_factory_mock:
            config = ColBERTConfig.load_from_checkpoint(model_name_or_path)
            config.doc_maxlen = self.max_length
            config.query_maxlen = self.max_length
            model = Checkpoint(name=model_name_or_path, colbert_config=config)
            model.query_tokenizer = ColBERTQueryTokenizer(config)
            return model

    def chunk2text(self, chunk: Chunk) -> str:
        return self.title_body_separator.concat([chunk.doc_summary, chunk.text])

    @torch.no_grad()
    def encode_queries(
        self, queries: List[Query], batch_size_query: int, show_progress_bar=True
    ) -> torch.Tensor:
        model: Checkpoint = self.model
        query_texts = [query.text for query in queries]
        qembs = model.queryFromText(queries=query_texts, bsize=batch_size_query)
        return qembs

    @torch.no_grad()
    def encode_documents(
        self,
        documents: List[Document],
        batch_size_document: Optional[int] = None,
        batch_size_chunk: Optional[int] = None,
    ) -> EmbeddingsWithMask:
        model: Checkpoint = self.model
        chunk_texts = [
            self.chunk2text(chunk) for doc in documents for chunk in doc.chunks
        ]
        chunk_embs, chunk_lengths = model.docFromText(
            docs=chunk_texts,
            bsize=batch_size_chunk,
            keep_dims="flatten",
            showprogress=False,
        )
        with torch.cuda.device(self.model.device):
            chunk_embs, cmask = StridedTensor(
                chunk_embs, chunk_lengths, use_gpu=True
            ).as_padded_tensor()  # (nchunks, chunk_length, hdim), (nchunks, chunk_lenght, 1)
        return EmbeddingsWithMask(embeddings=chunk_embs, mask=cmask)


class EncoderType(str, Enum):

    single_vector = "single_vector"
    colbert = "colbert"

    def __call__(self) -> Type[TextEncoder]:
        name2class = {
            EncoderType.single_vector: SingleVectorEncoder,
            EncoderType.colbert: ColBERTEncoder,
        }
        if self not in name2class:
            raise NotImplementedError
        return name2class[self]
