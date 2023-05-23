from typing import Dict, List
from dapr.inference.base import BaseExperiment, Config
from dapr.models.dm import rcls_x_rdls
from dapr.models.encoding import SimilarityFunction
from dapr.models.evaluation import EvaluationOutput, RetrievalLevel


class BM25DocBM25Chunk(BaseExperiment):
    """Q2C: BM25 on chunks + BM25 on documents; Q2D: BM25 on documents."""

    def score(self, cfg: Config) -> List[EvaluationOutput]:
        evaluator = self.build_evaluator(cfg=cfg)
        bm25 = self.build_bm25_retriever()
        rcls = bm25.retrieve_chunk_lists(**self.build_retrieval_kwargs(evaluator))
        rdls = bm25.retrieve_document_lists(**self.build_retrieval_kwargs(evaluator))

        approach_rcls_iterator = rcls_x_rdls(
            queries=evaluator.queries,
            rcls=rcls,
            rdls=rdls,
            similarity_function1=SimilarityFunction.dot_product,
            similarity_function2=SimilarityFunction.dot_product,
            name1="bm25_chunk",
            name2="bm25_doc",
            topk=cfg.experiment.topk,
        )

        eouts = []
        for approach, rcls in approach_rcls_iterator:
            report_chunk = evaluator(
                retrieved=rcls, level=RetrievalLevel.chunk, report_prefix=approach
            )
            eouts.append(report_chunk)
        report_doc = evaluator(retrieved=rdls, level=RetrievalLevel.document)
        eouts.append(report_doc)
        return eouts


if __name__ == "__main__":
    BM25DocBM25Chunk().run()

# nohup python -m dadpr.inference.bm25_doc_bm25_chunk +dataset=nq experiment.wandb=True > bm25_doc_bm25_chunk-nq.log&
# nohup python -m dadpr.inference.bm25_doc_bm25_chunk +dataset=genomics experiment.wandb=True > bm25_doc_bm25_chunk-genomics.log &
# nohup python -m dadpr.inference.bm25_doc_bm25_chunk +dataset=msmarco experiment.wandb=True > bm25_doc_bm25_chunk-msmarco.log &
