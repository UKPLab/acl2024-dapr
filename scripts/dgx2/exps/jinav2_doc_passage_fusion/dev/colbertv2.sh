export CUDA_VISIBLE_DEVICES="1,2,4,9"
dataset="MSMARCO"
export DATA_DIR="data"
export DATASET_PATH="$DATA_DIR/$dataset"
export CLI_ARGS="
--data_dir=$DATASET_PATH
--passage_results="$(ls exps/passage_only/colbertv2/data_dir_data/MSMARCO/split_dev/topk_1000/per_device_eval_batch_size_32/fp16_True/*/ranking_results.txt|head -1)"
--split=dev
"
export OUTPUT_DIR=$(python -m dapr.exps.jinav2_doc_passage_fusion.args.colbertv2 $CLI_ARGS)
mkdir -p $OUTPUT_DIR
export LOG_PATH="$OUTPUT_DIR/logging.log"
echo "Logging file path: $LOG_PATH"
torchrun --nproc_per_node=4 --master_port=29511 -m dapr.exps.jinav2_doc_passage_fusion.colbertv2 $CLI_ARGS
# setsid nohup torchrun --nproc_per_node=4 --master_port=29511 -m dapr.exps.jinav2_doc_passage_fusion.colbertv2 $CLI_ARGS > $LOG_PATH &