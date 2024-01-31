from clddp.retriever import Retriever, RetrieverConfig, Pooling, SimilarityFunction
from clddp.dm import Separator


class ColBERTv2(Retriever):
    def __init__(
        self, query_max_length: int = 150, passage_max_length: int = 512
    ) -> None:
        config = RetrieverConfig(
            query_model_name_or_path="colbert-ir/colbertv2.0",
            shared_encoder=True,
            sep=Separator.blank,
            pooling=Pooling.no_pooling,
            similarity_function=SimilarityFunction.maxsim,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
        )
        super().__init__(config)
