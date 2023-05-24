# Document-Aware Passage Retrieval (DAPR)

DAPR is a benchmark for document-aware passage retrieval: given a (large) collection of documents, relevant passages within these documents for a given query are required to be returned. 

A key focus of DAPR is forcing/encouraging retrieval systems to utilize the document-level context which surrounds the relevant passages. An example is shown below:

<img src='imgs/motivative-example.png' width='300'>

> In this example, the query asks for a musician or a group who has ever played at a certain venue. However, the gold relevant passage mentions only the reference noun, "the venue" but its actual name, "the Half Moon, Putney". The model thus needs to explore the context from belonging document of the passage, which in this case means coreference resolution.

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
# import os  # Please apply for the official COLIEE data and paste its Google drive IDs below:
# os.environ["COLIEE_TASK1_TRAIN_FILES"] = "<GOOGLE DRIVE ID of file task1_train_files_2023.zip>"
# os.environ["COLIEE_TASK2_TRAIN_FILES"] = "<GOOGLE DRIVE ID of file task2_train_files_2023.zip>"
# os.environ["COLIEE_TASK2_TRAIN_LABELS"] = "<GOOGLE DRIVE ID of file  task2_train_labels_2023.json>"
# from dapr.hydra_schemas.dataset import COLIEEConfig

dataset = NaturalQuestionsConfig()()
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
> Note that the original datasets of COLIEE are only available for application. Please apply for the data following [the official guide](https://sites.ualberta.ca/~rabelo/COLIEE2023/#:~:text=Memorandum%20for%20Tasks%201%20and/or%202%20(Case%20law%20competition)) and set up the Google drive IDs to the environment variables: `COLIEE_TASK1_TRAIN_FILES`, `COLIEE_TASK2_TRAIN_FILES` and `COLIEE_TASK2_TRAIN_LABELS` before running the code.

For evaluation, an example is as follows:
```python
from dadpr.models.evaluation import LongDocumentEvaluator
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
from dadpr.models.evaluation import LongDocumentEvaluator
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
evaluation_scores = evaluator(retrieved=retrieved, level=RetrievalLevel.chunk).summary
print(evaluation_scores)
```
An example of the BM25 baseline is available:
```
bash bm25.sh
```
> It requires JDK (openjdk>=11). One can install it via conda by `conda install openjdk=11`.
## Data
The pre-build data for `NaturalQuestions`, `MSMARCO`, `MIRACL` and `Genomics` are also available: https://public.ukp.informatik.tu-darmstadt.de/kwang/dapr/v1/