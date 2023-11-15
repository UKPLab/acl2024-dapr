from typing import Dict, List
from dapr.hydra_schemas.retrieval import BM25Config, RetrieverConfig
from dapr.inference.base import BaseExperiment, Config
from dapr.models.dm import RetrievedDocumentList, rcls_x_rdls, rdls_x_rdls
from dapr.models.encoding import SimilarityFunction
from dapr.models.evaluation import EvaluationOutput, RetrievalLevel
from dapr.models.retrieval.base import DocumentMethod
from dapr.models.retrieval.bm25 import BM25Retriever


class BM25DocNeuralMaxP(BaseExperiment):
    """Q2C: BM25 on documents + neural on chunks; Q2D: BM25 on documents + neural on chunks w/ MaxP."""

    def score(self, cfg: Config) -> List[EvaluationOutput]:
        evaluator = self.build_evaluator(cfg=cfg)
        bm25 = self.build_bm25_retriever()
        rdls_bm25 = bm25.retrieve_document_lists(
            **self.build_retrieval_kwargs(evaluator)
        )

        neural_retriever = self.build_neural_retriever(
            cfg, doc_method=DocumentMethod.max_p
        )
        rcls_neural = neural_retriever.retrieve_chunk_lists(
            **self.build_retrieval_kwargs(evaluator)
        )
        rdls_max_p = [rcl.max_p() for rcl in rcls_neural]

        # Fusion
        approach_rcls_iterator = rcls_x_rdls(
            queries=evaluator.queries,
            rcls=rcls_neural,
            rdls=rdls_bm25,
            similarity_function1=neural_retriever.query_encoder.similarity_function,
            similarity_function2=SimilarityFunction.dot_product,
            name1="neural",
            name2="bm25",
            topk=cfg.experiment.topk,
        )
        approach_rdls_iterator = rdls_x_rdls(
            queries=evaluator.queries,
            rdls1=rdls_bm25,
            rdls2=rdls_max_p,
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
            # RDLs (C2D):
            eouts.append(
                evaluator(
                    retrieved=[RetrievedDocumentList.from_rcl(rcl) for rcl in rcls],
                    level=RetrievalLevel.document,
                    report_prefix=f"{approach}-c2d",
                )
            )
        for approach, rdls in approach_rdls_iterator:
            report_doc = evaluator(
                retrieved=rdls, level=RetrievalLevel.document, report_prefix=approach
            )
            eouts.append(report_doc)
        return eouts


if __name__ == "__main__":
    BM25DocNeuralMaxP().run()

# export CUDA_VISIBLE_DEVICES=6; nohup python -m dapr.inference.bm25_doc_neural_max_p +dataset=nq +retriever=nq-distilbert-base-v1 experiment.wandb=True > nq-bm25_doc_neural_max_p.log &
# export CUDA_VISIBLE_DEVICES=9; nohup python -m dapr.inference.bm25_doc_neural_max_p +dataset=genomics +retriever=nq-distilbert-base-v1 experiment.wandb=True > genomics-bm25_doc_neural_max_p.log &
# export CUDA_VISIBLE_DEVICES=10; nohup python -m dapr.inference.bm25_doc_neural_max_p +dataset=msmarco +retriever=nq-distilbert-base-v1 experiment.wandb=True > msmarco-bm25_doc_neural_max_p.log &
