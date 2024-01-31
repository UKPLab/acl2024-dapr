export CUDA_VISIBLE_DEVICES="1,2,4,9"
datasets=( "ConditionalQA" "MSMARCO" "NaturalQuestions" "Genomics" "MIRACL" )
passage_results_paths=(
    "$(ls exps/passage_only/colbertv2/data_dir_data/ConditionalQA/split_test/topk_1000/per_device_eval_batch_size_32/fp16_True/*/ranking_results.txt|head -1)"
    "$(ls exps/passage_only/colbertv2/data_dir_data/MSMARCO/split_test/topk_1000/per_device_eval_batch_size_32/fp16_True/*/ranking_results.txt|head -1)"
    "$(ls exps/passage_only/colbertv2/data_dir_data/NaturalQuestions/split_test/topk_1000/per_device_eval_batch_size_32/fp16_True/*/ranking_results.txt|head -1)"
    "$(ls exps/passage_only/colbertv2/data_dir_data/Genomics/split_test/topk_1000/per_device_eval_batch_size_32/fp16_True/*/ranking_results.txt|head -1)"
    "$(ls exps/passage_only/colbertv2/data_dir_data/MIRACL/split_test/topk_1000/per_device_eval_batch_size_32/fp16_True/*/ranking_results.txt|head -1)"
)
for i in {0..4}
do
    dataset=${datasets[$i]}
    passage_results_path=${passage_results_paths[$i]}
    export DATA_DIR="data"
    export DATASET_PATH="$DATA_DIR/$dataset"
    export CLI_ARGS="
    --data_dir=$DATASET_PATH
    --passage_results=$passage_results_path
    --report_passage_weight=0.8
    "
    export OUTPUT_DIR=$(python -m dapr.exps.jinav2_doc_passage_fusion.args.colbertv2 $CLI_ARGS)
    mkdir -p $OUTPUT_DIR
    export LOG_PATH="$OUTPUT_DIR/logging.log"
    echo "Logging file path: $LOG_PATH"
    torchrun --nproc_per_node=4 --master_port=29502 -m dapr.exps.jinav2_doc_passage_fusion.colbertv2 $CLI_ARGS
done