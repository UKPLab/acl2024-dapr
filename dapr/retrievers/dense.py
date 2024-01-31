from clddp.retriever import Retriever, RetrieverConfig, Pooling, SimilarityFunction
from clddp.dm import Separator
from transformers import PreTrainedModel, AutoModel


class RetroMAE(Retriever):
    def __init__(self) -> None:
        config = RetrieverConfig(
            query_model_name_or_path="Shitao/RetroMAE_BEIR",
            shared_encoder=True,
            sep=Separator.bert_sep,
            pooling=Pooling.cls,
            similarity_function=SimilarityFunction.dot_product,
            query_max_length=512,
            passage_max_length=512,
        )
        super().__init__(config)


class DRAGONPlus(Retriever):
    def __init__(self) -> None:
        config = RetrieverConfig(
            query_model_name_or_path="facebook/dragon-plus-query-encoder",
            passage_model_name_or_path="facebook/dragon-plus-context-encoder",
            shared_encoder=False,
            sep=Separator.blank,
            pooling=Pooling.cls,
            similarity_function=SimilarityFunction.dot_product,
            query_max_length=512,
            passage_max_length=512,
        )
        super().__init__(config)


class JinaV2(Retriever):
    def __init__(self, max_length: int = 8192) -> None:
        config = RetrieverConfig(
            query_model_name_or_path="jinaai/jina-embeddings-v2-base-en",
            shared_encoder=True,
            sep=Separator.blank,
            pooling=Pooling.mean,
            similarity_function=SimilarityFunction.cos_sim,
            # max_length=512,  # Use this for paragraph-level retrieval
            query_max_length=max_length,
            passage_max_length=max_length,
        )
        super().__init__(config)

    @staticmethod
    def load_checkpoint(
        model_name_or_path: str, config: RetrieverConfig
    ) -> PreTrainedModel:
        return AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)

    # def encode(
    #     self, encoder: PreTrainedModel, texts: List[str], batch_size: int
    # ) -> torch.Tensor:
    #     return encoder.encode(
    #         texts,
    #         batch_size=batch_size,
    #         convert_to_tensor=True,
    #         convert_to_numpy=False,
    #         show_progress_bar=False,
    #     )
