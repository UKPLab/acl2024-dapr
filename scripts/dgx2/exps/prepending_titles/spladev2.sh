export NCCL_DEBUG="INFO"
export CUDA_VISIBLE_DEVICES="5,6,7,8"

datasets=( "ConditionalQA" "MSMARCO" "NaturalQuestions" "Genomics" "MIRACL" )
for dataset in ${datasets[@]}
do
    export DATA_DIR="data"
    export DATASET_PATH="$DATA_DIR/$dataset"
    export CLI_ARGS="
    --data_dir=$DATASET_PATH
    "
    export OUTPUT_DIR=$(python -m dapr.exps.prepending_titles.args.spladev2 $CLI_ARGS)
    mkdir -p $OUTPUT_DIR
    export LOG_PATH="$OUTPUT_DIR/logging.log"
    echo "Logging file path: $LOG_PATH"
    torchrun --nproc_per_node=4 --master_port=29502 -m dapr.exps.prepending_titles.spladev2 $CLI_ARGS > $LOG_PATH
done