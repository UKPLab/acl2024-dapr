from typing import Dict, List
from dapr.models.dm import RetrievedDocumentList

from dapr.hydra_schemas.retrieval import RetrieverConfig
from dapr.inference.base import BaseExperiment, Config
from dapr.models.evaluation import EvaluationOutput, RetrievalLevel


class SummarizedContextMaxP(BaseExperiment):
    """Q2C: Neural on (summarized context + chunk)'s; Q2D: Neural on (summarized context + chunk)'s -> MaxP."""

    def score(self, cfg: Config) -> List[EvaluationOutput]:
        annotator = self.build_annotator(cfg)
        evaluator = self.build_evaluator(cfg=cfg, annotator=annotator)
        retrieval_kwargs = self.build_retrieval_kwargs(evaluator)
        neural_retriever = self.build_neural_retriever(cfg)
        rcls = neural_retriever.retrieve_chunk_lists(**retrieval_kwargs)
        rdls_max_p = [rcl.max_p() for rcl in rcls]

        report_chunk = evaluator(retrieved=rcls, level=RetrievalLevel.chunk)
        report_doc = evaluator(retrieved=rdls_max_p, level=RetrievalLevel.document)
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
    SummarizedContextMaxP().run()

# export CUDA_VISIBLE_DEVICES=6; nohup python -m dapr.inference.summarized_context_max_p +dataset=nq +retriever=nq-distilbert-base-v1 +annotator=topic_rank experiment.wandb=True > nq-summarized_context_max_p-topic_rank.log &
# export CUDA_VISIBLE_DEVICES=9; nohup python -m dapr.inference.summarized_context_max_p +dataset=genomics +retriever=nq-distilbert-base-v1 +annotator=topic_rank experiment.wandb=True > genomics-summarized_context_max_p-topic_rank.log &
# export CUDA_VISIBLE_DEVICES=10; nohup python -m dapr.inference.summarized_context_max_p +dataset=msmarco +retriever=nq-distilbert-base-v1 +annotator=topic_rank experiment.wandb=True > msmarco-summarized_context_max_p-topic_rank.log &

# export CUDA_VISIBLE_DEVICES=6; nohup python -m dapr.inference.summarized_context_max_p +dataset=nq +retriever=nq-distilbert-base-v1 +annotator=lead experiment.wandb=True > nq-summarized_context_max_p-lead.log &
# export CUDA_VISIBLE_DEVICES=9; nohup python -m dapr.inference.summarized_context_max_p +dataset=genomics +retriever=nq-distilbert-base-v1 +annotator=lead experiment.wandb=True > genomics-summarized_context_max_p-lead.log &
# export CUDA_VISIBLE_DEVICES=10; nohup python -m dapr.inference.summarized_context_max_p +dataset=msmarco +retriever=nq-distilbert-base-v1 +annotator=lead experiment.wandb=True > msmarco-summarized_context_max_p-lead.log &

# export CUDA_VISIBLE_DEVICES=6; nohup python -m dapr.inference.summarized_context_max_p +dataset=nq +retriever=nq-distilbert-base-v1 +annotator=title experiment.wandb=True > nq-summarized_context_max_p-title.log &
# export CUDA_VISIBLE_DEVICES=9; nohup python -m dapr.inference.summarized_context_max_p +dataset=genomics +retriever=nq-distilbert-base-v1 +annotator=title experiment.wandb=True > genomics-summarized_context_max_p-title.log &
# export CUDA_VISIBLE_DEVICES=10; nohup python -m dapr.inference.summarized_context_max_p +dataset=msmarco +retriever=nq-distilbert-base-v1 +annotator=title experiment.wandb=True > msmarco-summarized_context_max_p-title.log &

# export CUDA_VISIBLE_DEVICES=6; nohup python -m dapr.inference.summarized_context_max_p +dataset=nq +retriever=nq-distilbert-base-v1 +annotator=empty experiment.wandb=True > nq-summarized_context_max_p-empty.log &
# export CUDA_VISIBLE_DEVICES=9; nohup python -m dapr.inference.summarized_context_max_p +dataset=genomics +retriever=nq-distilbert-base-v1 +annotator=empty experiment.wandb=True > genomics-summarized_context_max_p-empty.log &
# export CUDA_VISIBLE_DEVICES=10; nohup python -m dapr.inference.summarized_context_max_p +dataset=msmarco +retriever=nq-distilbert-base-v1 +annotator=empty experiment.wandb=True > msmarco-summarized_context_max_p-empty.log &
