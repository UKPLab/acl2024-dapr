from dapr.exps.jinav2_doc_passage_fusion.args.dragon_plus import (
    DRAGONPlusJinaV2DocPassageFusionArguments,
)
from dapr.exps.jinav2_doc_passage_fusion.shared_pipeline import (
    run_jina_doc_passage_fusion,
)

if __name__ == "__main__":
    run_jina_doc_passage_fusion(
        arguments_class=DRAGONPlusJinaV2DocPassageFusionArguments,
        passage_retriever_name="dragon_plus",
    )
