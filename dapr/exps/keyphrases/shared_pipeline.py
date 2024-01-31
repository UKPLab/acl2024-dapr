from dataclasses import dataclass
import json
import logging
import os
from typing import Dict, Iterable, Optional, TypedDict
from clddp.dm import LabeledQuery, Passage, RetrievalDataset, RetrievedPassageIDList
from clddp.retriever import Retriever
from clddp.utils import is_device_zero
from dapr.datasets.dm import LoadedData
from ...dataloader import DAPRDataConfig
from dapr.exps.keyphrases.args.base import KeyphrasesArguments
from clddp.search import search
from dapr.dataloader import DAPRDataConfig, DAPRDataLoader
from clddp.evaluation import RetrievalEvaluator
import ujson


class KeyphrasesRow(TypedDict):
    doc_id: str
    doc_summary: str


@dataclass
class KeyphrasesDAPRDataConfig(DAPRDataConfig):
    keyphrases_path: Optional[str] = None


class KeyphrasesDAPRDataLoader(DAPRDataLoader):
    def __init__(self, config: KeyphrasesDAPRDataConfig) -> None:
        self.config = config
        self.did2kps: Dict[str, str] = {}
        self.pid2did: Dict[str, str] = {}

    def collection_iter_fn(self, data: LoadedData) -> Iterable[Passage]:
        for psg in super().collection_iter_fn(data):
            kps = self.did2kps[self.pid2did[psg.passage_id]]
            yield Passage(passage_id=psg.passage_id, text=psg.text, title=kps)

    def load_data(self, progress_bar: bool) -> RetrievalDataset:
        logging.info("Loading keyphrases")
        with open(self.config.keyphrases_path) as f:
            for line in f:
                line_dict: KeyphrasesRow = ujson.loads(line)
                self.did2kps[line_dict["doc_id"]] = line_dict["doc_summary"]

        self.pid2did = self.get_pid2did(progress_bar)
        return super().load_data(progress_bar)


def run_keyphrases(
    args: KeyphrasesArguments, retriever: Retriever, retriever_name: str
) -> None:
    # Actually the same as passage_only. And the difference relies on the data
    if is_device_zero():
        args.dump_arguments()
    dataset = KeyphrasesDAPRDataLoader(
        KeyphrasesDAPRDataConfig(
            data_name_or_path=args.data_dir,
            keyphrases_path=args.keyphrases_path,
            titled=True,  # Actually the keyphrases here
        )
    ).load_data(progress_bar=is_device_zero())
    labeled_queries = dataset.get_labeled_queries(args.split)
    queries = LabeledQuery.get_unique_queries(labeled_queries)
    retrieved = search(
        retriever=retriever,
        collection_iter=dataset.collection_iter,
        collection_size=dataset.collection_size,
        queries=queries,
        topk=args.topk,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        fp16=args.fp16,
    )
    if is_device_zero():
        evaluator = RetrievalEvaluator(eval_dataset=dataset, split=args.split)
        report = evaluator(retrieved)
        freport = os.path.join(args.output_dir, "metrics.json")
        with open(freport, "w") as f:
            json.dump(report, f, indent=4)
        logging.info(f"Saved evaluation metrics to {freport}.")
        franked = os.path.join(args.output_dir, "ranking_results.txt")
        RetrievedPassageIDList.dump_trec_csv(
            retrieval_results=retrieved, fpath=franked, system=retriever_name
        )
        logging.info(f"Saved ranking results to {franked}.")
