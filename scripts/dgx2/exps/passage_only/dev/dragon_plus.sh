export NCCL_DEBUG="INFO"
export CUDA_VISIBLE_DEVICES="2,9,10,11"

# datasets=( "ConditionalQA" "MSMARCO" "NaturalQuestions" "Genomics" "MIRACL" )
# for dataset in ${datasets[@]}
# do
#     export DATA_DIR="data"
#     export DATASET_PATH="$DATA_DIR/$dataset"
#     export CLI_ARGS="
#     --data_dir=$DATASET_PATH
#     "
#     export OUTPUT_DIR=$(python -m dapr.exps.passage_only.args.dragon_plus $CLI_ARGS)
#     mkdir -p $OUTPUT_DIR
#     export LOG_PATH="$OUTPUT_DIR/logging.log"
#     echo "Logging file path: $LOG_PATH"
#     # nohup torchrun --nproc_per_node=4 --master_port=29501 -m dapr.exps.passage_only.dragon_plus $CLI_ARGS > $LOG_PATH &
#     torchrun --nproc_per_node=4 --master_port=29502 -m dapr.exps.passage_only.dragon_plus $CLI_ARGS > $LOG_PATH
# done
# nohup bash scripts/dgx2/exps/passage_only/dragon_plus.sh > passage_only_dragon_plus2.log &

# torchrun --nproc_per_node=1 --master_port=29501 -m dapr.exps.passage_only.dragon_plus $CLI_ARGS


export DATA_DIR="data"
export DATASET_PATH="$DATA_DIR/MSMARCO"
export CLI_ARGS="
--data_dir=$DATASET_PATH
--split=dev
"
export OUTPUT_DIR=$(python -m dapr.exps.passage_only.args.dragon_plus $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
setsid nohup torchrun --nproc_per_node=4 --master_port=29503 -m dapr.exps.passage_only.dragon_plus $CLI_ARGS > $LOG_PATH &
# torchrun --nproc_per_node=4 --master_port=29502 -m dapr.exps.passage_only.dragon_plus $CLI_ARGS > $LOG_PATH