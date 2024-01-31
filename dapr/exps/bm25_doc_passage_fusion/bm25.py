from dapr.exps.bm25_doc_passage_fusion.args.bm25 import (
    BM25DocBM25PassageFusionArguments,
)
from dapr.exps.bm25_doc_passage_fusion.shared_pipeline import (
    run_bm25_doc_passage_fusion,
)

if __name__ == "__main__":
    run_bm25_doc_passage_fusion(
        arguments_class=BM25DocBM25PassageFusionArguments,
        passage_retriever_name="bm25",
    )
