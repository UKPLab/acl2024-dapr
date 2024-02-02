export CUDA_VISIBLE_DEVICES="0,1,2,3"
dataset="MSMARCO"
export DATA_DIR="data"
export DATASET_PATH="$DATA_DIR/$dataset"
export CLI_ARGS="
--data_dir=$DATASET_PATH
--split=dev
"
export OUTPUT_DIR=$(python -m dapr.exps.bm25_doc_passage_hierarchy.args.spladev2 $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
setsid nohup torchrun --nproc_per_node=4 --master_port=29501 -m dapr.exps.bm25_doc_passage_hierarchy.spladev2 $CLI_ARGS > $LOG_PATH &