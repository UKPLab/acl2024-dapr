# Document-Aware Passage Retrieval (DAPR)

Paper draft: https://arxiv.org/abs/2305.13915 (work in progress)

DAPR is a benchmark for document-aware passage retrieval: given a (large) collection of documents, relevant passages within these documents for a given query are required to be returned. 

A key focus of DAPR is forcing/encouraging retrieval systems to utilize the document-level context which surrounds the relevant passages. An example is shown below:

<img src='imgs/motivative-example.png' width='300'>

> In this example, the query asks for a musician or a group who has ever played at a certain venue. However, the gold relevant passage mentions only the reference noun, "the venue" but its actual name, "the Half Moon, Putney". The model thus needs to explore the context from the belonging document of the passage, which in this case means coreference resolution.

## Installation:
Python>=3.8 is required. Run this installation script below:
```bash
pip install git+https://github.com/kwang2049/dapr.git
```
For the optional usage of BM25, please install JDK (`openjdk`>=11). One can install it via conda:
```bash
conda install openjdk=11
```


## Usage
### Building/loading data
```python
from dapr.datasets.conditionalqa import ConditionalQA
from dapr.datasets.nq import NaturalQuestions
from dapr.datasets.genomics import Genomics
from dapr.datasets.miracl import MIRACL
from dapr.datasets.msmarco import MSMARCO
from dapr.datasets.dm import LoadedData

# Build the data on the fly: (this will save the data to ./data/ConditionalQA)
data = ConditionalQA().loaded_data  # Also the same for NaturalQuestions, etc.
# data = LoadedData.from_dump("data/ConditionalQA")  # Load the pre-built data (please download it from https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/ConditionalQA)

# Iterate over the corpus:
for doc in data.corpus_iter_fn():
    doc.doc_id
    doc.title
    for chunk in doc.chunks:
        chunk.chunk_id
        chunk.text
        chunk.belonging_doc.doc_id

# Iterate over the labeled queries (of the test split):
for labeled_query in data.labeled_queries_test:
    labeled_query.query.query_id
    labeled_query.query.text
    for judged_chunk in labeled_query.judged_chunks:
        judged_chunk.chunk.chunk_id
        judged_chunk.chunk.text
        judged_chunk.chunk.belonging_doc.doc_id
```

### Evaluation 
```python
from typing import Dict
from dapr.retrievers.dense import DRAGONPlus
from dapr.datasets.conditionalqa import ConditionalQA
from clddp.dm import Query, Passage
import torch
import pytrec_eval
import numpy as np

# Load data:
data = ConditionalQA().loaded_data

# Encode queries and passages:
retriever = DRAGONPlus()
retriever.eval()
queries = [
    Query(query_id=labeled_query.query.query_id, text=labeled_query.query.text)
    for labeled_query in data.labeled_queries_test
]
passages = [
    Passage(passage_id=chunk.chunk_id, text=chunk.text)
    for doc in data.corpus_iter_fn()
    for chunk in doc.chunks
]
query_embeddings = retriever.encode_queries(queries)
with torch.no_grad():  # Takes around a minute on a V100 GPU
    passage_embeddings, passage_mask = retriever.encode_passages(passages)

# Calculate the similarities and keep top-K:
similarity_scores = torch.matmul(
    query_embeddings, passage_embeddings.t()
)  # (query_num, passage_num)
topk = torch.topk(similarity_scores, k=10)
topk_values: torch.Tensor = topk[0]
topk_indices: torch.LongTensor = topk[1]
topk_value_lists = topk_values.tolist()
topk_index_lists = topk_indices.tolist()

# Run evaluation with pytrec_eval:
retrieval_scores: Dict[str, Dict[str, float]] = {}
for query_i, (values, indices) in enumerate(zip(topk_value_lists, topk_index_lists)):
    query_id = queries[query_i].query_id
    retrieval_scores.setdefault(query_id, {})
    for value, passage_i in zip(values, indices):
        passage_id = passages[passage_i].passage_id
        retrieval_scores[query_id][passage_id] = value
qrels: Dict[str, Dict[str, int]] = {
    labeled_query.query.query_id: {
        judged_chunk.chunk.chunk_id: judged_chunk.judgement
        for judged_chunk in labeled_query.judged_chunks
    }
    for labeled_query in data.labeled_queries_test
}
evaluator = pytrec_eval.RelevanceEvaluator(
    query_relevance=qrels, measures=["ndcg_cut_10"]
)
query_performances: Dict[str, Dict[str, float]] = evaluator.evaluate(retrieval_scores)
ndcg = np.mean([score["ndcg_cut_10"] for score in query_performances.values()])
print(ndcg)  # 0.21796083196880855
```

### Reproducing experiment results
All the experiment scripts are available at [scripts/dgx2/exps](scripts/dgx2/exps). For example, one can evaluate the DRAGON+ retriever in a passage-only manner like this:
```bash
# scripts/dgx2/exps/passage_only/dragon_plus.sh
export NCCL_DEBUG="INFO"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

datasets=( "ConditionalQA" )
# datasets=( "ConditionalQA" "MSMARCO" "NaturalQuestions" "Genomics" "MIRACL" )
for dataset in ${datasets[@]}
do
    export DATA_DIR="data"
    export DATASET_PATH="$DATA_DIR/$dataset"
    export CLI_ARGS="
    --data_dir=$DATASET_PATH
    "
    export OUTPUT_DIR=$(python -m dapr.exps.passage_only.args.dragon_plus $CLI_ARGS)
    mkdir -p $OUTPUT_DIR
    export LOG_PATH="$OUTPUT_DIR/logging.log"
    echo "Logging file path: $LOG_PATH"
    torchrun --nproc_per_node=4 --master_port=29501 -m dapr.exps.passage_only.dragon_plus $CLI_ARGS > $LOG_PATH
done
```

## Pre-Built Data
The building processes above require relative large memory for the large datasets. The loading part after this data building is cheap though (the collections will be loaded on the fly via Python generators). The budgets are listed below (with 12 multi-processes):
| Dataset    | Memory |  Time |
| -------- | ------- | ------- |
| NaturalQuestions  | 25.6GB    | 39min    |
| Genomics | 18.3GB     |25min    |
| MSMARCO    | 102.9GB    |3h    |
| MIRACL    | 69.7GB    |1h30min    |
| ConditionalQA    |  <1GB | <1min |

To bypass this, one can also download the pre-built data: 
```bash
mkdir data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/NaturalQuestions/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/MSMARCO/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/Genomics/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/MIRACL/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data/ConditionalQA/ -P ./data
```
The data are also available at the Huggingface datasets: https://huggingface.co/datasets/kwang2049/dapr.

## Citation
If you use the code/data, feel free to cite our publication [DAPR: A Benchmark on Document-Aware Passage Retrieval](https://arxiv.org/abs/2305.13915):
```bibtex 
@article{wang2023dapr,
    title = "DAPR: A Benchmark on Document-Aware Passage Retrieval",
    author = "Kexin Wang and Nils Reimers and Iryna Gurevych", 
    journal= "arXiv preprint arXiv:2305.13915",
    year = "2023",
    url = "https://arxiv.org/abs/2305.13915",
}
```

Contact person and main contributor: [Kexin Wang](https://kwang2049.github.io/), kexin.wang.2049@gmail.com

[https://www.ukp.tu-darmstadt.de/](https://www.ukp.tu-darmstadt.de/)

[https://www.tu-darmstadt.de/](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Updates
- Feb. 06. 2023
    - Rename the folder name from v3 into data in the fileserver.
    - Uploaded other data like coference-resolution results, extracted keyphrases, and the experiment results.
    - Created the HF datasets.
    - Refactored the experiment code, aligning with the new paper version.
- Nov. 16, 2023
    - New version of data uploaded to https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/data
    - Replaced COLIEE with ConditionalQA
        - ConditionalQA has two sub-versions here: (1) ConditionalQA, the original dataset; (2) CleanedConditionalQA whose html tags are removed.
    - The MSMARCO dataset now segments the documents by keeping the labeled paragraphs while leaving the leftover parts as the other paragraphs.
        - For example, given the original unsegmented document text "11122222334444566" and if the labeled paragraphs are "22222" and "4444", then the segmentations will be ["111", "22222", "33", "4444", "566"].
        - We only do retrieval over the labeled paragraphs, which is specified by the attribute "candidate_chunk_ids" of each document object.
    - We now use only the specific version of the ColBERT package, as the latest one has some unknown issue.
