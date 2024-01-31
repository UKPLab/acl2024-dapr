export NCCL_DEBUG="INFO"
export CUDA_VISIBLE_DEVICES="5,6,7,8"

datasets=( "coref/ConditionalQA" "coref/MSMARCO" "coref/NaturalQuestions" "coref/Genomics" "coref/MIRACL" )
# datasets=( "coref/ConditionalQA" )
for dataset in ${datasets[@]}
do
    export DATA_DIR="data"
    export DATASET_PATH="$DATA_DIR/$dataset"
    export CLI_ARGS="
    --data_dir=$DATASET_PATH
    "
    export OUTPUT_DIR=$(python -m dapr.exps.coref.args.spladev2 $CLI_ARGS)
    mkdir -p $OUTPUT_DIR
    export LOG_PATH="$OUTPUT_DIR/logging.log"
    echo "Logging file path: $LOG_PATH"
    torchrun --nproc_per_node=4 --master_port=29509 -m dapr.exps.coref.spladev2 $CLI_ARGS > $LOG_PATH
done