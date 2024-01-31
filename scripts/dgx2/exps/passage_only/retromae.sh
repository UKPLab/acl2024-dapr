export CUDA_VISIBLE_DEVICES="0,1"

dataset="ConditionalQA"
export DATA_DIR="data"
export DATASET_PATH="$DATA_DIR/$dataset"
export CLI_ARGS="
--data_dir=$DATASET_PATH
"
export OUTPUT_DIR=$(python -m dapr.exps.passage_only.args.retromae $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
nohup torchrun --nproc_per_node=2 --master_port=29507 -m dapr.exps.passage_only.retromae $CLI_ARGS > $LOG_PATH &