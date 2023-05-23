from colbert.modeling.colbert import colbert_score
from dapr.hydra_schemas.retrieval import ColBERTV2Config, RetroMAEBEIRConfig
from dapr.models.encoding import ColBERTEncoder, SingleVectorEncoder, maxsim
from tests.conftest import DatasetFixture
from tests.integration.test_datasets import build_nq
import torch


def test_single_vector_encoder_loading() -> None:
    SingleVectorEncoder(
        model_name_or_path=RetroMAEBEIRConfig.query_model,
        similarity_function=RetroMAEBEIRConfig.similarity_function,
        pooling=RetroMAEBEIRConfig.pooling,
        max_length=RetroMAEBEIRConfig.max_length,
        max_nchunks=RetroMAEBEIRConfig.max_nchunks,
        title_body_separator=RetroMAEBEIRConfig.title_body_separator,
        chunk_separator=RetroMAEBEIRConfig.chunk_separator,
    )


def test_colbert_encoder_loading() -> None:
    ColBERTEncoder(
        model_name_or_path=ColBERTV2Config.query_model,
        similarity_function=ColBERTV2Config.similarity_function,
        pooling=ColBERTV2Config.pooling,
        max_length=ColBERTV2Config.max_length,
        max_nchunks=ColBERTV2Config.max_nchunks,
        title_body_separator=ColBERTV2Config.title_body_separator,
        chunk_separator=ColBERTV2Config.chunk_separator,
    )


def test_maxsim_correct_implemention(
    nq_resource: DatasetFixture, cache_root_dir: str
) -> None:
    dataset = build_nq(data_path=nq_resource.data_path, cache_root_dir=cache_root_dir)
    colbert = ColBERTEncoder(
        model_name_or_path=ColBERTV2Config.query_model,
        similarity_function=ColBERTV2Config.similarity_function,
        pooling=ColBERTV2Config.pooling,
        max_length=ColBERTV2Config.max_length,
        max_nchunks=ColBERTV2Config.max_nchunks,
        title_body_separator=ColBERTV2Config.title_body_separator,
        chunk_separator=ColBERTV2Config.chunk_separator,
    )
    colbert.to("cuda")
    docs = list(dataset.loaded_data.corpus_iter_fn())
    queries = [lq.query for lq in dataset.loaded_data.labeled_queries_test]
    qembs = colbert.encode_queries(queries=queries, batch_size_query=16)
    cembs_with_mask = colbert.encode_documents(documents=docs, batch_size_chunk=16)
    scores = maxsim(
        qembs=qembs, cembs=cembs_with_mask.embeddings, cmask=cembs_with_mask.mask
    )
    for i in range(len(queries)):
        score = colbert_score(
            qembs[i : i + 1], cembs_with_mask.embeddings, cembs_with_mask.mask
        )
        assert torch.allclose(scores[i], score)
