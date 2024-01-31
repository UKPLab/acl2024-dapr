from dapr.exps.bm25_doc_passage_fusion.args.dragon_plus import (
    DRAGONPlusBM25DocPassageFusionArguments,
)
from dapr.exps.bm25_doc_passage_fusion.shared_pipeline import (
    run_bm25_doc_passage_fusion,
)

if __name__ == "__main__":
    run_bm25_doc_passage_fusion(
        arguments_class=DRAGONPlusBM25DocPassageFusionArguments,
        passage_retriever_name="dragon_plus",
    )
