from __future__ import annotations
from datetime import datetime
from enum import Enum
from io import BufferedWriter
from itertools import chain
from multiprocessing.context import SpawnProcess
import pickle
import random
import sys
import tempfile
import hashlib
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)
from typing_extensions import ParamSpec
from zlib import crc32
from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.utils import _split_into_batches
from colbert.utils.utils import torch_load_dnn
from filelock import FileLock
import psutil
import requests
import tqdm
import shutil
import zipfile
import numpy as np
from nltk.tokenize import TextTilingTokenizer, sent_tokenize
from torch.utils.checkpoint import get_device_states, set_device_states
import git

import logging
from logging import StreamHandler

import torch
from transformers import (
    PreTrainedModel,
    LongformerModel,
    LongformerConfig,
    LongformerForMaskedLM,
    BertModel,
    BertPreTrainedModel,
    BertConfig,
    AutoTokenizer,
    BertTokenizer,
)
import multiprocessing as mp
from multiprocessing.queues import Queue

NINF = -1e4
SOFT_ZERO = 1e-4

T = TypeVar("T")
P = ParamSpec("P")
I = TypeVar("I")
O = TypeVar("O")


def sha256(text: str) -> str:
    """Generate a sha256 code for the input text."""
    return hashlib.sha256(text.encode()).hexdigest()


def md5(texts: Iterable[str]) -> str:
    md5 = hashlib.md5()
    for text in texts:
        md5.update(text.encode())
    return md5.hexdigest()


def hash_model(model: torch.nn.Module) -> str:
    """Generate a md5 code for a PyTorch model."""
    hasher = hashlib.md5()
    for p in model.parameters():
        hasher.update(pickle.dumps(p.cpu().detach().numpy()))
    return hasher.hexdigest()


def download(url: str, fto: str, chunk_size=1024):
    """Download resource."""
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    try:
        with open(fto, "wb") as fd, tqdm.tqdm(
            desc=f"Downloading to {fto}",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            for data in r.iter_content(chunk_size=chunk_size):
                size = fd.write(data)
                bar.update(size)
    except Exception as e:
        os.remove(fto)
        raise e


def unzip_file(fpath: str, output_dir: str):
    """
    Unzip a zip file.

    :param fpath: The path to the zip file.
    :param output_dir: Should be the path where the resource resides.
    """
    assert fpath.endswith(".zip")
    with zipfile.ZipFile(fpath, "r") as zip:
        zip.extractall(path=output_dir)


def download_and_cache(url: str, unzip=False):
    """Download a resource into a local temporary cached folder."""
    fname = os.path.basename(url)
    if unzip:
        assert fname.endswith(".zip")
        resource_name = fname.replace(".zip", "")
    else:
        resource_name = fname

    hash = sha256(url)
    local_dir = os.path.join(tempfile.gettempdir(), hash)
    local_path = os.path.join(local_dir, resource_name)
    if os.path.exists(local_path):
        return local_path

    os.makedirs(local_dir)
    try:
        fdownloaded = os.path.join(local_dir, fname)
        download(url, fdownloaded)
        if unzip:
            unzip_file(fdownloaded, local_dir)
        return local_path
    except Exception as e:
        shutil.rmtree(local_dir)
        raise e


def nlines(fpath: str) -> int:
    """Count how many lines in a file."""
    with open(fpath, "r") as f:
        return sum(1 for _ in f)


def tqdm_ropen(fpath: str, desc: str = None) -> Iterator[str]:
    """tqdm + open with r mode."""
    if desc is None:
        desc = f"Loading from {fpath}"

    with tqdm.tqdm(open(fpath, "r"), desc, nlines(fpath)) as f:
        for line in f:
            yield line


def randomly_split_by_number(
    data: List[T], number: int, seed=42
) -> Tuple[List[T], List[T]]:
    """Randomly split the data by specifying number."""
    random_state = random.Random(seed)
    positions = list(range(len(data)))
    sampled = random_state.sample(positions, number)
    sampled_set = set(sampled)
    left = list(filter(lambda pos: pos not in sampled_set, positions))
    data_array = np.array(data)
    return data_array[sampled].tolist(), data_array[left].tolist()


def randomly_split_by_ratio(
    data: List[T], ratio: float, seed=42
) -> Tuple[List[T], List[T]]:
    """Randomly split the data by specifying ratio."""
    return randomly_split_by_number(data, int(len(data) * ratio), seed)


def set_logger_format() -> None:
    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in root_logger.handlers:
        if isinstance(handler, StreamHandler):
            handler.setFormatter(formatter)


def cache_to_disk(cache_path: str, tmp: bool = False):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def new_func(*args: P.args, **kwargs: P.kwargs) -> T:
            logger = logging.getLogger(__name__)
            cpath = cache_path
            if tmp:
                cpath = os.path.join(tempfile.gettempdir(), sha256(cache_path))

            with FileLock(cpath + ".lock"):
                if os.path.exists(cpath):
                    logger.info(f"Found existing cache {cpath}")
                    with open(cpath, "rb") as f:
                        result = pickle.load(f)
                        logger.info(f"Loaded from cache {cpath}")
                else:
                    logger.info("No cache found. Running the task for the first time")
                    result = func(*args, **kwargs)
                    with open(cpath, "wb") as f:
                        pickle.dump(result, f)
                        logger.info(f"Saved cache to {cpath}")
                return result

        return new_func

    return decorator


def get_commit_hash() -> str:
    """Return the HEAD commit hash."""
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def time_stamp() -> str:
    """Return the current time (s)."""
    return datetime.now().strftime(r"%d-%m-%Y-%H-%M-%S")


def switch(enable: bool):
    """A switch decorator to control whether to run a function."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def new_func(*args: P.args, **kwargs: P.kwargs) -> T:
            if enable:
                return func(*args, **kwargs)
            else:
                pass

        return new_func

    return decorator


def hash_string_into_float(s: str, encoding: str = "utf-8"):
    """
    Hash a string into a float number between 0 and 1.

    Reference: https://stackoverflow.com/questions/40351791/how-to-hash-strings-into-a-float-in-01
    """
    bytes = s.encode(encoding)
    return float(crc32(bytes) & 0xFFFFFFFF) / 2**32


def extend(model: PreTrainedModel, ntimes=2) -> LongformerModel:
    """Extend the max. seq. length of a longformer model by `ntimes` times."""
    assert ntimes >= 1
    if type(model) == LongformerForMaskedLM:
        model: LongformerForMaskedLM
        model = model.longformer

    embedding_weights = model.embeddings.position_embeddings.weight
    model.embeddings.position_embeddings.weight = torch.nn.Parameter(
        torch.cat([embedding_weights for _ in range(ntimes)])
    )
    config: LongformerConfig = model.config
    config.max_position_embeddings = len(model.embeddings.position_embeddings.weight)
    return model


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.
    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors: torch.Tensor):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def concat_and_chunk(
    sequences: Iterable[List[T]], marked: Iterable[bool], chunk_size: int
) -> Iterator[Tuple[List[T], bool]]:
    sequences_concatenated = list(chain(*sequences))
    marked_broadcasted = list(
        chain(*map(lambda args: [args[1]] * len(sequences[args[0]]), enumerate(marked)))
    )

    for b in range(0, len(sequences_concatenated), chunk_size):
        e = b + chunk_size
        yield sequences_concatenated[b:e], any(marked_broadcasted[b:e])


def texttiling(text: str, w: int = 10, k: int = 2) -> List[str]:
    """Segmenting text into topically separate paragraphs via TextTiling."""
    DOUBLE_NEWLINES = "\n\n"
    tt = TextTilingTokenizer(w=w, k=k)
    try:
        tokenized: List[str] = tt.tokenize(DOUBLE_NEWLINES.join(sent_tokenize(text)))
    except ValueError:
        return [text]
    return [seg.replace(DOUBLE_NEWLINES, " ").strip() for seg in tokenized]


class Separator(str, Enum):
    bert_sep = "[SEP]"
    roberta_sep = "</s>"
    blank = " "
    empty = ""

    def concat(self, texts: Iterator[Optional[str]]) -> str:
        """Concatenate two pieces of texts with the separation symbol."""
        return self.join(filter(lambda text: text is not None, texts))


def build_mask_by_lengths(lengths: List[int], device: torch.device) -> torch.Tensor:
    # Create a tensor of sequence lengths
    seq_lengths = torch.tensor(lengths)  # (bsz,)
    max_length = seq_lengths.max()
    seq_lengths = seq_lengths.unsqueeze(-1).expand(len(seq_lengths), max_length)

    # Create a tensor of ones with the same size as the sequence lengths
    mask = torch.ones(seq_lengths.size())

    # Create a tensor containing a range of values from 0 to the maximum sequence length
    values = torch.arange(0, seq_lengths.max()).unsqueeze(0).expand_as(seq_lengths)

    # Use the torch.where() function to select elements from the tensor of ones
    # based on the values in the tensor of range values
    mask = torch.where(values < seq_lengths, mask, torch.zeros_like(mask))
    return mask.to(device)


def build_init_args_with_kwargs_and_default(
    init_fn: Callable, kwargs_specified: Dict[str, Any], kwargs_default: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build kwargs for the initialization function with the specified kwargs and default kwargs.
    The arguments will be the intersection between `kwargs_specified` and `kwargs_default`.
    If an argument is None in `kwargs_specified`, it should be found in `kwargs_default`.
    """
    init_args = set(init_fn.__code__.co_varnames) - {"self"}
    kwargs_default = dict(filter(lambda kv: kv[0] in init_args, kwargs_default.items()))
    kwargs = {}
    for k in set(kwargs_specified) & set(kwargs_default):
        if kwargs_specified[k] is None:
            assert k in kwargs_default
            kwargs[k] = kwargs_default[k]
        else:
            kwargs[k] = kwargs_specified[k]
    return kwargs


def remove_and_shift_left_2d(x: torch.Tensor, value: float) -> torch.Tensor:
    # x: (bsz, seq_len)
    result = x.new_full(x.shape, value)
    bsz, seq_length = x.shape
    index = (
        torch.arange(start=0, end=seq_length, device=x.device).unsqueeze(0).expand_as(x)
    )
    mask = x.ne(value)
    seq_lengths = mask.sum(dim=-1, keepdim=True)
    index_bool = (
        index.masked_fill(index >= seq_lengths, 0)
        .masked_fill(index < seq_lengths, 1)
        .bool()
    )
    result[index_bool] = x[mask]
    return result


def build_mask_by_beginnings_and_endings(
    beginnings: torch.Tensor, endings: torch.Tensor, device: torch.device
) -> torch.Tensor:
    # beginnings: (bsz,)
    # endings: (bsz,)
    shifts = endings - beginnings
    assert sum(shifts >= 0) == len(shifts)
    beginnings = beginnings - beginnings.min()
    endings = beginnings + shifts
    bsz = shifts.shape[0]
    length_max = endings.max().long()
    index = (
        torch.arange(0, length_max, device=device).squeeze(0).expand(bsz, length_max)
    )
    left_mask = ((index - beginnings.unsqueeze(-1) + 1) > 0).bool()
    right_mask = ((endings.unsqueeze(-1) - index) > 0).bool()
    mask = left_mask & right_mask
    return mask


class Multiprocesser:
    def __init__(self, nprocs: int = 16) -> None:
        self.nprocs = nprocs
        self.ps: Optional[List[SpawnProcess]] = None
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _worker(qin: Queue, qout: Queue, func: Callable[[I], O]) -> None:
        logger = logging.getLogger(__name__)
        while True:
            i, data_point = qin.get()
            try:
                processed = func(data_point)
            except Exception as e:
                logger.error(str(e))
                raise e
            qout.put((i, processed))

    def watch_alive(self) -> None:
        for p in self.ps:
            if not p.is_alive():
                for p in self.ps:
                    p.terminate()
                self.logger.info("Exit due to error in subprocess")
                sys.exit()

    def _process_chunk(
        self, qout: Queue, chunk: List[int], progress_bar: tqdm.tqdm
    ) -> Dict[int, O]:
        result_dict = {}
        for _ in chunk:
            self.watch_alive()
            i, processed = qout.get()
            result_dict[i] = processed
            progress_bar.update(1)
        return result_dict

    def run(
        self,
        data: Iterable[I],
        func: Callable[[I], O],
        desc: str,
        total: Optional[int] = None,
        chunk_size: int = 5000,
    ) -> List[O]:
        ctx = mp.get_context("spawn")
        qin = ctx.Queue(chunk_size)
        qout = ctx.Queue(chunk_size // self.nprocs)
        self.ps = [
            ctx.Process(target=self._worker, args=(qin, qout, func), daemon=True)
            for _ in range(self.nprocs)
        ]
        try:
            for p in tqdm.tqdm(self.ps, desc="Starting processes"):
                p.start()

            result_dict = {}
            data_iter = iter(data)
            base = 0
            chunk = []
            pbar = tqdm.tqdm(total=total, desc=desc)
            while True:
                try:
                    qin.put((base + len(chunk), next(data_iter)))
                    chunk.append(None)
                    if (len(chunk) + 1) % chunk_size == 0:
                        result_dict.update(
                            self._process_chunk(
                                qout=qout, chunk=chunk, progress_bar=pbar
                            )
                        )
                        base += len(chunk)
                        chunk = []
                except StopIteration:
                    result_dict.update(
                        self._process_chunk(qout=qout, chunk=chunk, progress_bar=pbar)
                    )
                    base += len(chunk)
                    chunk = []
                    break

        finally:
            for p in self.ps:
                p.kill()

        result_list = [result_dict[i] for i in range(base)]
        return result_list


def memory_used(location: Optional[str]) -> None:
    logger = logging.getLogger(__name__)
    pid = os.getpid()
    mem = psutil.Process(pid).memory_info()[0] / 2**30
    mem_string = f"Memory used: {mem:.2f}GB"
    if location:
        mem_string += f" (location: {location})"
    logger.info(mem_string)


class Pooling(str, Enum):
    cls = "cls"
    mean = "mean"
    splade = "splade"
    sum = "sum"
    max = "max"
    no_pooling = "no_pooling"

    def __call__(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pooling: (bsz, seq_len, hdim) -> (bsz, hdim) or return the input without pooling."""

        if self == Pooling.cls:
            return token_embeddings[:, 0:1].sum(dim=1)
        elif self == Pooling.mean:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        elif self == Pooling.sum:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1)
        elif self == Pooling.max:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            token_embeddings[input_mask_expanded == 0] = (
                -1e9
            )  # Set padding tokens to large negative value
            return torch.max(token_embeddings, 1)[0]
        elif self == Pooling.splade:
            pooled: torch.Tensor = getattr(
                torch.max(
                    torch.log(1 + torch.relu(token_embeddings))
                    * attention_mask.unsqueeze(-1),
                    dim=1,
                ),
                "values",
            )
            return pooled
        elif self == Pooling.no_pooling:
            return token_embeddings
        else:
            return NotImplementedError


def download(
    url,
    temp_file: BufferedWriter,
    proxies=None,
    resume_size=0,
    headers=None,
    cookies=None,
    timeout=100.0,
    max_retries=0,
    desc=None,
) -> None:
    chunk_size = 1024 * 1024  # HF's datasets use 1024, which is too small
    r = requests.get(url, stream=True, timeout=100)
    total = int(r.headers.get("Content-Length", 0))
    with tqdm.tqdm(desc=desc, total=total, unit="B") as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = temp_file.write(data)
            bar.update(size)


class HF_ColBERT(BertPreTrainedModel):
    """
    Shallow wrapper around HuggingFace transformers. All new parameters should be defined at this level.

    This makes sure `{from,save}_pretrained` and `init_weights` are applied to new parameters correctly.
    """

    _keys_to_ignore_on_load_unexpected = [r"cls"]

    def __init__(self, config: BertConfig, colbert_config: ColBERTConfig):
        super().__init__(config)

        self.config = config
        self.dim = colbert_config.dim
        self.linear = torch.nn.Linear(
            config.hidden_size, colbert_config.dim, bias=False
        )
        setattr(self, self.base_model_prefix, BertModel(config))
        self.init_weights()

    @property
    def LM(self) -> BertModel:
        base_model_prefix = getattr(self, "base_model_prefix")
        return getattr(self, base_model_prefix)

    @classmethod
    def from_pretrained(
        cls: Type[HF_ColBERT], name_or_path: str, colbert_config: ColBERTConfig
    ) -> HF_ColBERT:
        if name_or_path.endswith(".dnn"):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get("arguments", {}).get("model", "bert-base-uncased")

            obj: BertPreTrainedModel = super().from_pretrained(
                base, state_dict=dnn["model_state_dict"], colbert_config=colbert_config
            )
            setattr(obj, "base", base)
            return obj

        obj = super().from_pretrained(name_or_path, colbert_config=colbert_config)
        setattr(obj, "base", name_or_path)
        return obj

    @staticmethod
    def raw_tokenizer_from_pretrained(name_or_path: str) -> BertTokenizer:
        if name_or_path.endswith(".dnn"):
            dnn = torch_load_dnn(name_or_path)
            base = dnn.get("arguments", {}).get("model", "bert-base-uncased")

            obj: BertTokenizer = AutoTokenizer.from_pretrained(base)
            setattr(obj, "base", base)
            return obj

        obj = AutoTokenizer.from_pretrained(name_or_path)
        setattr(obj, "base", name_or_path)
        return obj


class ColBERTQueryTokenizer(QueryTokenizer):
    def tensorize(
        self,
        batch_text: List[str],
        bsize: Optional[int] = None,
        context: Optional[List[str]] = None,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        assert type(batch_text) in [list, tuple], type(batch_text)

        # add placehold for the [Q] marker
        batch_text = [". " + x for x in batch_text]

        obj = self.tok(
            batch_text,
            # padding="max_length",
            # truncation=True,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.query_maxlen,
        )

        ids, mask = obj["input_ids"], obj["attention_mask"]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == self.pad_token_id] = self.mask_token_id

        if context is not None:
            assert len(context) == len(batch_text), (len(context), len(batch_text))

            obj_2 = self.tok(
                context,
                padding="longest",
                truncation=True,
                return_tensors="pt",
                max_length=self.background_maxlen,
            )

            ids_2, mask_2 = (
                obj_2["input_ids"][:, 1:],
                obj_2["attention_mask"][:, 1:],
            )  # Skip the first [SEP]

            ids = torch.cat((ids, ids_2), dim=-1)
            mask = torch.cat((mask, mask_2), dim=-1)

        if self.config.attend_to_mask_tokens:
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        if self.used is False:
            self.used = True

            firstbg = (context is None) or context[0]

            print()
            print(
                "#> QueryTokenizer.tensorize(batch_text[0], batch_background[0], bsize) =="
            )
            print(f"#> Input: {batch_text[0]}, \t\t {firstbg}, \t\t {bsize}")
            print(f"#> Output IDs: {ids[0].size()}, {ids[0]}")
            print(f"#> Output Mask: {mask[0].size()}, {mask[0]}")
            print()

        return ids, mask
