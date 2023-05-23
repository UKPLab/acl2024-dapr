from typing import Dict, Iterable, List, Tuple
from dapr.hydra_schemas.retrieval import BM25Config, RetrieverConfig
from dapr.inference.base import BaseExperiment, Config
from dapr.models.dm import RetrievedChunkList, rcls_x_rcls
from dapr.models.encoding import SimilarityFunction
from dapr.models.evaluation import EvaluationOutput, RetrievalLevel
from dapr.models.retrieval.bm25 import BM25Retriever


class BM25ChunkNeuralChunkMaxP(BaseExperiment):
    """Q2C: BM25 on chunks + neural on chunks; Q2D: (BM25 on chunks + neural on chunks) -> w/ MaxP."""

    def score(self, cfg: Config) -> List[EvaluationOutput]:
        evaluator = self.build_evaluator(cfg=cfg)
        bm25 = self.build_bm25_retriever()
        rcls_bm25 = bm25.retrieve_chunk_lists(**self.build_retrieval_kwargs(evaluator))
        neural_retriever = self.build_neural_retriever(cfg)
        rcls_neural = neural_retriever.retrieve_chunk_lists(
            **self.build_retrieval_kwargs(evaluator)
        )

        # Fusion
        approach_rcls_iterator: Iterable[
            Tuple[str, List[RetrievedChunkList]]
        ] = rcls_x_rcls(
            queries=evaluator.queries,
            rcls1=rcls_bm25,
            rcls2=rcls_neural,
            similarity_function1=SimilarityFunction.dot_product,
            similarity_function2=neural_retriever.query_encoder.similarity_function,
            name1="bm25",
            name2="neural",
            topk=cfg.experiment.topk,
        )
        eouts = []
        for approach, rcls in approach_rcls_iterator:
            # RCLs:
            report_chunk = evaluator(
                retrieved=rcls, level=RetrievalLevel.chunk, report_prefix=approach
            )
            eouts.append(report_chunk)
            # RDLs:
            rdls = [rcl.max_p() for rcl in rcls]
            report_doc = evaluator(
                retrieved=rdls, level=RetrievalLevel.document, report_prefix=approach
            )
            eouts.append(report_doc)
        return eouts


if __name__ == "__main__":
    BM25ChunkNeuralChunkMaxP().run()

# export CUDA_VISIBLE_DEVICES=6; nohup python -m dadpr.inference.bm25_chunk_neural_chunk_max_p +dataset=nq +retriever=nq-distilbert-base-v1 experiment.wandb=True > nq-bm25_chunk_neural_chunk_max_p.log &
# export CUDA_VISIBLE_DEVICES=9; nohup python -m dadpr.inference.bm25_chunk_neural_chunk_max_p +dataset=genomics +retriever=nq-distilbert-base-v1 experiment.wandb=True > genomics-bm25_chunk_neural_chunk_max_p.log &
# export CUDA_VISIBLE_DEVICES=10; nohup python -m dadpr.inference.bm25_chunk_neural_chunk_max_p +dataset=msmarco +retriever=nq-distilbert-base-v1 experiment.wandb=True > msmarco-bm25_chunk_neural_chunk_max_p.log &
