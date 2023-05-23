import os
import shutil
from typing import Tuple
from unittest import mock
from dapr.datasets.base import BaseDataset
from dapr.datasets.dm import Chunk, Document, JudgedChunk, Query
from dapr.datasets.nq import NaturalQuestions
from dapr.datasets.msmarco import MSMARCO
from tests.conftest import DatasetFixture


def build_nq(data_path: str, cache_root_dir: str) -> NaturalQuestions:
    mock_download = mock.patch.object(NaturalQuestions, "_download", return_value=None)
    mock_fdev = mock.patch.object(
        NaturalQuestions, "fdev", os.path.join(data_path, "dev.tsv")
    )
    mock_ftrain = mock.patch.object(
        NaturalQuestions, "ftrain", os.path.join(data_path, "train.tsv")
    )
    with mock_download, mock_fdev, mock_ftrain:
        dataset = NaturalQuestions(
            resource_path=data_path,
            nheldout=None,
            cache_root_dir=cache_root_dir,
            nprocs=2,
        )
    return dataset


def build_msmarco(data_path: str, cache_root_dir: str) -> MSMARCO:
    mock_download = mock.patch.object(NaturalQuestions, "_download", return_value=None)
    mock_fdev_qna = mock.patch.object(
        MSMARCO,
        "fdev_qna",
        os.path.join(data_path, "dev_v2.1.json"),
    )
    mock_ftrain_qna = mock.patch.object(
        MSMARCO,
        "ftrain_qna",
        os.path.join(data_path, "train_v2.1.json"),
    )
    mock_fcorpus = mock.patch.object(
        MSMARCO,
        "fcorpus",
        os.path.join(data_path, "msmarco-docs.tsv"),
    )
    mock_ftitles_pranking = mock.patch.object(
        MSMARCO,
        "ftitles_pranking",
        os.path.join(data_path, "para.title.txt"),
    )
    mock_fdev_queries_pranking = mock.patch.object(
        MSMARCO,
        "fdev_queries_pranking",
        os.path.join(data_path, "queries.dev.tsv"),
    )
    mock_fdev_qrels_small_pranking = mock.patch.object(
        MSMARCO,
        "fdev_qrels_small_pranking",
        os.path.join(data_path, "qrels.dev.small.tsv"),
    )
    with mock_download, mock_fdev_qna, mock_ftrain_qna, mock_fcorpus, mock_ftitles_pranking, mock_fdev_queries_pranking, mock_fdev_qrels_small_pranking:
        dataset = MSMARCO(
            resource_path=data_path,
            nheldout=None,
            cache_root_dir=cache_root_dir,
            nprocs=2,
        )
    return dataset


def test_load_msmarco(msmarco_resource: DatasetFixture, cache_root_dir: str):
    try:
        dataset = build_msmarco(
            data_path=msmarco_resource.data_path, cache_root_dir=cache_root_dir
        )
        assert dataset.loaded_data.meta_data["ndocs"] == msmarco_resource.ndocs
        assert len(dataset.loaded_data.labeled_queries_train) == msmarco_resource.ntrain
        assert len(dataset.loaded_data.labeled_queries_dev) == msmarco_resource.ndev
        assert len(dataset.loaded_data.labeled_queries_test) == msmarco_resource.ntest
    finally:
        if os.path.exists(cache_root_dir):
            shutil.rmtree(cache_root_dir)
    return dataset


def test_load_nq(nq_resource: DatasetFixture, cache_root_dir: str):
    # Given:
    nq_resource

    # When:
    try:
        dataset = build_nq(
            data_path=nq_resource.data_path, cache_root_dir=cache_root_dir
        )
        # Then:
        assert dataset.loaded_data.meta_data["ndocs"] == nq_resource.ndocs
        assert len(dataset.loaded_data.labeled_queries_train) == nq_resource.ntrain
        assert len(dataset.loaded_data.labeled_queries_dev) == nq_resource.ndev
        assert len(dataset.loaded_data.labeled_queries_test) == nq_resource.ntest
    finally:
        if os.path.exists(cache_root_dir):
            shutil.rmtree(cache_root_dir)


def test_msmarco_resegment(
    msmarco_resource: DatasetFixture, cache_root_dir: str
) -> None:
    try:
        msmarco = build_msmarco(
            data_path=msmarco_resource.data_path, cache_root_dir=cache_root_dir
        )
        msmarco.nwords_list = [2, 3, 4]
        doc = Document(doc_id="D123", chunks=[], title=None)
        chk0 = Chunk(
            chunk_id="D123-0",
            text="This is a sentence.",
            doc_summary=None,
            belonging_doc=doc,
        )
        chk1 = Chunk(
            chunk_id="D123-1",
            text="Just another sentence. And blablabla.",
            doc_summary=None,
            belonging_doc=doc,
        )
        chk2 = Chunk(
            chunk_id="D123-2",
            text="Just another another sentence. What do you think",
            doc_summary=None,
            belonging_doc=doc,
        )
        chk3 = Chunk(
            chunk_id="D123-3",
            text="End of the document.",
            doc_summary=None,
            belonging_doc=doc,
        )
        doc.chunks = [chk0, chk1, chk2, chk3]
        jchks = [
            JudgedChunk(
                query=Query(query_id="test-1", text="test=1"), chunk=chk2, judgement=1
            ),
            JudgedChunk(
                query=Query(query_id="test-0", text="test=0"), chunk=chk1, judgement=1
            ),
        ]
        jchks_new = msmarco._resegement(document=doc, judged_chunks=jchks)
        for jchk, jchk_new in zip(jchks, jchks_new):
            assert jchk.query.text == jchk_new.query.text
            assert jchk.chunk.text == jchk_new.chunk.text
    finally:
        if os.path.exists(cache_root_dir):
            shutil.rmtree(cache_root_dir)
