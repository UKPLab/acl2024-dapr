from dapr.exps.bm25_doc_passage_fusion.args.colbertv2 import (
    CoLBERTv2BM25DocPassageFusionArguments,
)
from dapr.exps.bm25_doc_passage_fusion.shared_pipeline import (
    run_bm25_doc_passage_fusion,
)

if __name__ == "__main__":
    run_bm25_doc_passage_fusion(
        arguments_class=CoLBERTv2BM25DocPassageFusionArguments,
        passage_retriever_name="colbertv2",
    )
