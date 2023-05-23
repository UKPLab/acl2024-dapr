from typing import Dict, List
from dapr.hydra_schemas.retrieval import BM25Config, RetrieverConfig
from dapr.inference.base import BaseExperiment, Config
from dapr.models.dm import rcls_x_rdls, rdls_x_rdls
from dapr.models.encoding import SimilarityFunction
from dapr.models.evaluation import EvaluationOutput, RetrievalLevel
from dapr.models.retrieval.base import DocumentMethod
from dapr.models.retrieval.bm25 import BM25Retriever
from dapr.models.retrieval.neural import NeuralRetriever


class BM25NeuralFirstP(BaseExperiment):
    """Q2D: BM25 on documents + neural on chunks w/ FirstP; Q2C: BM25 on chunks + neural on chunks w/ FirstP."""

    def score(self, cfg: Config) -> List[EvaluationOutput]:
        evaluator = self.build_evaluator(cfg=cfg)
        bm25 = self.build_bm25_retriever()
        rdls_bm25 = bm25.retrieve_document_lists(
            **self.build_retrieval_kwargs(evaluator)
        )
        rcls_bm25 = bm25.retrieve_chunk_lists(**self.build_retrieval_kwargs(evaluator))

        neural_retriever = self.build_neural_retriever(
            cfg, doc_method=DocumentMethod.first_p
        )
        rdls_neural = neural_retriever.retrieve_document_lists(
            **self.build_retrieval_kwargs(evaluator)
        )

        # Fusion
        approach_rdls_iterator = rdls_x_rdls(
            queries=evaluator.queries,
            rdls1=rdls_bm25,
            rdls2=rdls_neural,
            similarity_function1=SimilarityFunction.dot_product,
            similarity_function2=neural_retriever.query_encoder.similarity_function,
            name1="bm25",
            name2="neural",
            topk=cfg.experiment.topk,
        )
        approach_rcls_iterator = rcls_x_rdls(
            queries=evaluator.queries,
            rcls=rcls_bm25,
            rdls=rdls_neural,
            similarity_function1=SimilarityFunction.dot_product,
            similarity_function2=neural_retriever.query_encoder.similarity_function,
            name1="bm25",
            name2="neural",
            topk=cfg.experiment.topk,
        )

        # Run evaluation
        eouts = []
        for approach, rcls in approach_rcls_iterator:
            report_chunk = evaluator(
                retrieved=rcls, level=RetrievalLevel.chunk, report_prefix=approach
            )
            eouts.append(report_chunk)
        for approach, rdls in approach_rdls_iterator:
            report_doc = evaluator(
                retrieved=rdls, level=RetrievalLevel.document, report_prefix=approach
            )
            eouts.append(report_doc)
        return eouts


if __name__ == "__main__":
    BM25NeuralFirstP().run()

# export CUDA_VISIBLE_DEVICES=6; nohup python -m dadpr.inference.bm25_neural_first_p +dataset=nq +retriever=nq-distilbert-base-v1 experiment.wandb=True > nq-bm25_neural_first_p.log &
# export CUDA_VISIBLE_DEVICES=9; nohup python -m dadpr.inference.bm25_neural_first_p +dataset=genomics +retriever=nq-distilbert-base-v1 experiment.wandb=True > genomics-bm25_neural_first_p.log &
# export CUDA_VISIBLE_DEVICES=10; nohup python -m dadpr.inference.bm25_neural_first_p +dataset=msmarco +retriever=nq-distilbert-base-v1 experiment.wandb=True > msmarco-bm25_neural_first_p.log &
