from dapr.exps.jinav2_doc_passage_fusion.args.colbertv2 import (
    CoLBERTv2JinaV2DocPassageFusionArguments,
)
from dapr.exps.jinav2_doc_passage_fusion.shared_pipeline import (
    run_jina_doc_passage_fusion,
)

if __name__ == "__main__":
    run_jina_doc_passage_fusion(
        arguments_class=CoLBERTv2JinaV2DocPassageFusionArguments,
        passage_retriever_name="colbertv2",
    )
