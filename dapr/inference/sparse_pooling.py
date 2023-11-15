from typing import Dict, List
from dapr.inference.base import BaseExperiment, Config
from dapr.models.dm import RetrievedDocumentList, rcls_x_rdls
from dapr.models.encoding import SimilarityFunction
from dapr.models.evaluation import EvaluationOutput, RetrievalLevel
from dapr.models.retrieval.base import PoolingDocumentMethods


class SparsePooling(BaseExperiment):
    """Q2C: pooled sparse representation on documents + sparse retrieval on chunks; Q2D: pooled sparse representation on documents."""

    def score(self, cfg: Config) -> List[EvaluationOutput]:
        evaluator = self.build_evaluator(cfg=cfg)
        assert cfg.retriever.doc_method in PoolingDocumentMethods
        neural_retriever = self.build_neural_retriever(cfg=cfg, doc_method=None)
        rcls = neural_retriever.retrieve_chunk_lists(
            **self.build_retrieval_kwargs(evaluator)
        )
        neural_retriever = self.build_neural_retriever(
            cfg, doc_method=cfg.retriever.doc_method
        )
        rdls = neural_retriever.retrieve_document_lists(
            **self.build_retrieval_kwargs(evaluator)
        )

        # Fusion
        approach_rcls_iterator = rcls_x_rdls(
            queries=evaluator.queries,
            rcls=rcls,
            rdls=rdls,
            similarity_function1=neural_retriever.query_encoder.similarity_function,
            similarity_function2=SimilarityFunction.dot_product,
            name1="neural",
            name2="bm25",
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
        report_doc = evaluator(retrieved=rdls, level=RetrievalLevel.document)
        eouts.append(report_doc)
        return eouts


if __name__ == "__main__":
    SparsePooling().run()

# export CUDA_VISIBLE_DEVICES=6; python -m dapr.inference.sparse_pooling +dataset=nq +retriever="nq-distilbert-base-v1" retriever.doc_method="pooling_max" experiment.wandb=True
# export CUDA_VISIBLE_DEVICES=6; python -m dapr.inference.sparse_pooling +dataset=nq +retriever="splade-cocondenser-ensembledistil" retriever.doc_method="pooling_max" experiment.wandb=True
