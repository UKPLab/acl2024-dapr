# Document-Aware Passage Retrieval (DAPR)

Paper draft: https://arxiv.org/abs/2305.13915 (work in progress)

DAPR is a benchmark for document-aware passage retrieval: given a (large) collection of documents, relevant passages within these documents for a given query are required to be returned. 

A key focus of DAPR is forcing/encouraging retrieval systems to utilize the document-level context which surrounds the relevant passages. An example is shown below:

<img src='imgs/motivative-example.png' width='300'>

> In this example, the query asks for a musician or a group who has ever played at a certain venue. However, the gold relevant passage mentions only the reference noun, "the venue" but its actual name, "the Half Moon, Putney". The model thus needs to explore the context from the belonging document of the passage, which in this case means coreference resolution.

## Usage
Python>=3.8 is required. Run this installation script below:
```bash
pip install git+https://github.com/kwang2049/dapr.git
```
And then the evaluation data can be accessed by building it on the fly (from their original sources):
```python
from dapr.hydra_schemas.dataset import NaturalQuestionsConfig
## Other datasets can be imported similarly:
# from dapr.hydra_schemas.dataset import MSMARCOConfig
# from dapr.hydra_schemas.dataset import GenomicsConfig
# from dapr.hydra_schemas.dataset import MIRACLConfig
# from dapr.hydra_schemas.dataset import CleanedConditionalQA

dataset = NaturalQuestionsConfig(cache_root_dir="data")()
for doc in dataset.loaded_data.corpus_iter_fn():
    for chunk in doc.chunks:
        (chunk.chunk_id, chunk.text)

for labeled_query in dataset.loaded_data.labeled_queries_test:
    for judged_chunk in labeled_query.judged_chunks:
        (
            labeled_query.query.query_id, 
            labeled_query.query.text, 
            judged_chunk.chunk.chunk_id, 
            judged_chunk.chunk.text, 
            judged_chunk.judgement
        )
```

For evaluation, an example is as follows:
```python
from dapr.models.evaluation import LongDocumentEvaluator
from dapr.datasets.dm import Split
from dapr.models.dm import RetrievalLevel, RetrievedChunkList, ScoredChunk
from dapr.hydra_schemas.dataset import NaturalQuestionsConfig

dataset = NaturalQuestionsConfig()()
evaluator = LongDocumentEvaluator(data=dataset.load_data, results_dir="results", split=Split.test)
retrieved = [
    RetrievedChunkList(query_id="query0", scored_chunks=[
        ScoredChunk(chunk_id="doc0-chunk0", doc_id="doc0", score=4.0),
        ScoredChunk(chunk_id="doc3-chunk5", doc_id="doc3", score=3.0),
        ScoredChunk(chunk_id="doc2-chunk4", doc_id="doc2", score=1.0),
    ]),
    RetrievedChunkList(query_id="query1", scored_chunks=[
        ScoredChunk(chunk_id="doc7-chunk6", doc_id="doc7", score=9.0),
        ScoredChunk(chunk_id="doc4-chunk3", doc_id="doc4", score=5.0),
        ScoredChunk(chunk_id="doc1-chunk0", doc_id="doc1", score=2.0),
    ]),
]
evaluation_scores = evaluator(retrieved=retrieved, level=RetrievalLevel.chunk).summary
print(evaluation_scores)
```
The evaluation for document retrieval is also available:
```python
from dapr.models.evaluation import LongDocumentEvaluator
from dapr.datasets.dm import Split
from dapr.models.dm import RetrievalLevel, RetrievedDocumentList, ScoredDocument
from dapr.hydra_schemas.dataset import NaturalQuestionsConfig

dataset = NaturalQuestionsConfig()()
evaluator = LongDocumentEvaluator(data=dataset.load_data, results_dir="results", split=Split.test)
retrieved = [
    RetrievedDocumentList(query_id="query0", scored_documents=[
        ScoredDocument(doc_id="doc0", score=4.0),
        ScoredDocument(doc_id="doc3", score=3.0),
        ScoredDocument(chunk_id="doc2-chunk4", doc_id="doc2", score=1.0),
    ]),
    RetrievedDocumentList(query_id="query1", scored_documents=[
        ScoredDocument(doc_id="doc7", score=9.0),
        ScoredDocument(doc_id="doc4", score=5.0),
        ScoredDocument(doc_id="doc1", score=2.0),
    ]),
]
evaluation_scores = evaluator(retrieved=retrieved, level=RetrievalLevel.document).summary
print(evaluation_scores)
```
An example of the BM25 baseline is available:
```
bash bm25.sh
```
> It requires JDK (openjdk>=11). One can install it via conda by `conda install openjdk=11`.
## Pre-Built Data
The building processes above require relative large memory for the large datasets. The loading part after this data building is cheap though (the collections will be loaded on the fly via Python generators). The budgets are listed below (with 12 multi-processes):
| Dataset    | Memory |  Time |
| -------- | ------- | ------- |
| NaturalQuestions  | 25.6GB    | 39min    |
| Genomics | 18.3GB     |25min    |
| MSMARCO    | 102.9GB    |3h    |
| MIRACL    | 69.7GB    |1h30min    |
| ConditionalQA    |   | |

To bypass this, one can also download the pre-built data for `NaturalQuestions`, `MSMARCO`, `MIRACL` and `Genomics`: 
```bash
mkdir data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/v3/NaturalQuestions/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/v3/MSMARCO/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/v3/Genomics/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/v3/MIRACL/ -P ./data
wget -r -np -nH --cut-dirs=3 https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/v3/ConditionalQA/ -P ./data
```

## Updates
- Nov. 16, 2023
    - New version of data uploaded to https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/v3
    - Replaced COLIEE with ConditionalQA
        - ConditionalQA has two sub-versions here: (1) ConditionalQA, the original dataset; (2) CleanedConditionalQA whose html tags are removed.
    - The MSMARCO dataset now segments the documents by keeping the labeled paragraphs while leaving the leftover parts as the other paragraphs.
        - For example, given the original unsegmented document text "11122222334444566" and if the labeled paragraphs are "22222" and "4444", then the segmentations will be ["111", "22222", "33", "4444", "566"].
        - We only do retrieval over the labeled paragraphs, which is specified by the attribute "candidate_chunk_ids" of each document object.
    - We now use only the specific version of the ColBERT package, as the latest one has some unknown issue.