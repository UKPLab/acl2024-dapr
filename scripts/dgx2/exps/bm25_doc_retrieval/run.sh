datasets=( "ConditionalQA" "MSMARCO" "NaturalQuestions" "Genomics" "MIRACL" )
for i in {0..4}
do
    dataset=${datasets[$i]}
    passage_results_path=${passage_results_paths[$i]}
    export DATA_DIR="data"
    export DATASET_PATH="$DATA_DIR/$dataset"
    export CLI_ARGS="
    --data_dir=$DATASET_PATH
    "
    export OUTPUT_DIR=$(python -m dapr.exps.bm25_doc_retrieval.args $CLI_ARGS)
    mkdir -p $OUTPUT_DIR
    export LOG_PATH="$OUTPUT_DIR/logging.log"
    echo "Logging file path: $LOG_PATH"
    python -m dapr.exps.bm25_doc_retrieval.pipeline $CLI_ARGS > $LOG_PATH
done