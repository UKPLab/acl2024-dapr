## Please set up these environment variables for building COLIEE:
# export COLIEE_TASK1_TRAIN_FILES="<GOOGLE DRIVE ID>"
# export COLIEE_TASK2_TRAIN_FILES="<GOOGLE DRIVE ID>"
# export COLIEE_TASK2_TRAIN_LABELS="<GOOGLE DRIVE ID>"

python -m dapr.inference.bm25 +dataset=nq

# dataset could be nq, msmarco, genomics, miracl and coliee.