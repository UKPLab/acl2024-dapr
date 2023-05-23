from dataclasses import dataclass
import os
import shutil
import tempfile

import pytest
import torch


@dataclass
class DatasetFixture:
    data_path: str
    ndocs: int
    ntrain: int
    ndev: int
    ntest: int


@pytest.fixture(name="nq_resource")
def nq_resource_fixture() -> DatasetFixture:
    return DatasetFixture(
        data_path="./sample-data/NaturalQuestions", ndocs=9, ntrain=3, ndev=3, ntest=3
    )


@pytest.fixture(name="msmarco_resource")
def msmarco_resource_fixture() -> DatasetFixture:
    return DatasetFixture(
        data_path="./sample-data/MSMARCO", ndocs=12, ntrain=6, ndev=3, ntest=3
    )


@pytest.fixture(name="results_dir")
def results_dir_fixture() -> str:
    return "results-pytest"


@pytest.fixture(name="cache_root_dir")
def cache_root_dir_fixture() -> str:
    return "data-pytest"


@pytest.fixture(name="distilbert_path", scope="session")
def distilbert_path_fixture() -> str:
    try:
        local_dir = tempfile.mkdtemp()

        torch.manual_seed(42)
        vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "the",
            "of",
            "and",
            "in",
            "to",
            "was",
            "he",
        ]
        vocab_file = os.path.join(local_dir, "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab))

        from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

        config = DistilBertConfig(
            vocab_size=len(vocab),
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=2,
            intermediate_size=2,
            max_position_embeddings=512,
        )

        bert = DistilBertModel(config)
        tokenizer = DistilBertTokenizer(vocab_file)

        bert.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        yield local_dir
    finally:
        shutil.rmtree(local_dir)
        print("Cleared temporary DistilBERT model")
