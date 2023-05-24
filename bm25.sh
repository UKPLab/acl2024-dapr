## Please set up these environment variables for building COLIEE:
# export COLIEE_TASK1_TRAIN_FILES="<GOOGLE DRIVE ID of file task1_train_files_2023.zip>"
# export COLIEE_TASK2_TRAIN_FILES="<GOOGLE DRIVE ID of file task2_train_files_2023.zip>"
# export COLIEE_TASK2_TRAIN_LABELS="<GOOGLE DRIVE ID of file  task2_train_labels_2023.json>"

python -m dapr.inference.bm25 +dataset=nq

# dataset could be nq, msmarco, genomics, miracl and coliee.