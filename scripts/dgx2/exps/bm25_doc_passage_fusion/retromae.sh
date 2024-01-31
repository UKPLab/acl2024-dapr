dataset="ConditionalQA"
export DATA_DIR="data"
export DATASET_PATH="$DATA_DIR/$dataset"
export CLI_ARGS="
--data_dir=$DATASET_PATH
--passage_results="$(ls exps/passage_only/retromae/data_dir_data/ConditionalQA/split_test/topk_1000/per_device_eval_batch_size_32/fp16_True/*/ranking_results.txt|head -1)"
"
export OUTPUT_DIR=$(python -m dapr.exps.bm25_doc_passage_fusion.args.retromae $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
nohup python -m dapr.exps.bm25_doc_passage_fusion.retromae $CLI_ARGS > $LOG_PATH &