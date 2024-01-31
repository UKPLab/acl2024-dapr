from clddp.retriever import Retriever, RetrieverConfig, Pooling, SimilarityFunction
from clddp.dm import Separator


class SPLADEv2(Retriever):
    def __init__(self) -> None:
        config = RetrieverConfig(
            query_model_name_or_path="naver/splade-cocondenser-ensembledistil",
            shared_encoder=True,
            sep=Separator.blank,
            pooling=Pooling.splade,
            similarity_function=SimilarityFunction.dot_product,
            query_max_length=512,
            passage_max_length=512,
        )
        super().__init__(config)
