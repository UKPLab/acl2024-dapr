from typing import Dict, List
from dapr.models.dm import RetrievedDocumentList
from dapr.hydra_schemas.retrieval import BM25Config
from dapr.inference.base import BaseExperiment, Config
from dapr.models.evaluation import EvaluationOutput, RetrievalLevel


class BM25(BaseExperiment):
    """Q2C: BM25 on chunks; Q2D: BM25 on documents."""

    def score(self, cfg: Config) -> List[EvaluationOutput]:
        evaluator = self.build_evaluator(cfg=cfg)
        bm25 = self.build_bm25_retriever()
        rcls = bm25.retrieve_chunk_lists(**self.build_retrieval_kwargs(evaluator))
        rdls = bm25.retrieve_document_lists(**self.build_retrieval_kwargs(evaluator))

        report_chunk = evaluator(retrieved=rcls, level=RetrievalLevel.chunk)
        report_doc = evaluator(retrieved=rdls, level=RetrievalLevel.document)
        eouts = [report_chunk, report_doc]
        # RDLs (C2D):
        eouts.append(
            evaluator(
                retrieved=[RetrievedDocumentList.from_rcl(rcl) for rcl in rcls],
                level=RetrievalLevel.document,
                report_prefix="c2d",
            )
        )
        return eouts


if __name__ == "__main__":
    BM25().run()

# nohup python -m dapr.inference.bm25 +dataset=nq experiment.wandb=True > bm25-nq.log&
# nohup python -m dapr.inference.bm25 +dataset=genomics experiment.wandb=True > bm25-genomics.log &
# nohup python -m dapr.inference.bm25 +dataset=msmarco experiment.wandb=True > bm25-msmarco.log &
