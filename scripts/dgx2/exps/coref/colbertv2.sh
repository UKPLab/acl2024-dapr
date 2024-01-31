export NCCL_DEBUG="INFO"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

datasets=( "coref/ConditionalQA" "coref/MSMARCO" "coref/NaturalQuestions" "coref/Genomics" "coref/MIRACL" )
for dataset in ${datasets[@]}
do
    export DATA_DIR="data"
    export DATASET_PATH="$DATA_DIR/$dataset"
    export CLI_ARGS="
    --data_dir=$DATASET_PATH
    "
    export OUTPUT_DIR=$(python -m dapr.exps.coref.args.colbertv2 $CLI_ARGS)
    mkdir -p $OUTPUT_DIR
    export LOG_PATH="$OUTPUT_DIR/logging.log"
    echo "Logging file path: $LOG_PATH"
    torchrun --nproc_per_node=4 --master_port=29503 -m dapr.exps.coref.colbertv2 $CLI_ARGS > $LOG_PATH
done