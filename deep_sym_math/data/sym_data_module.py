from pathlib import Path
import pytorch_lightning as pl
import os
from deep_sym_math.data.util import download_url
from deep_sym_math.constants import SYM_URLS, SPECIAL_WORDS, OPERATORS
import io
from torch.utils.data import DataLoader
from deep_sym_math.data.base_dataset import BaseDataset
from collections import OrderedDict
import sympy as sp
import torch

# Define file paths
DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"
RAW_DATA_DIRNAME = DATA_DIR / "raw_sym_datasets"
PROCESSED_DATA_DIRNAME = DATA_DIR / "symbolic_datasets"


class SymDataModule(pl.LightningDataModule):

    def __init__(self, tasks):
        super().__init__()
        self.batch_size = 32
        self.num_workers = 0  # Count of subprocesses to use for data loading
        self.tasks = tasks
        self.max_elements = 2
        self.int_base = 10

        # Symbols / Elements
        self.operators = sorted(list(OPERATORS.keys()))
        self.constants = ['pi', 'E']
        self.variables = OrderedDict({
            'x': sp.Symbol('x', real=True, nonzero=True),  # , positive=True
            'y': sp.Symbol('y', real=True, nonzero=True),  # , positive=True
            'z': sp.Symbol('z', real=True, nonzero=True),  # , positive=True
            't': sp.Symbol('t', real=True, nonzero=True),  # , positive=True
        })
        self.coefficients = OrderedDict(
            {f'a{i}': sp.Symbol(f'a{i}', real=True) for i in range(10)})
        self.functions = OrderedDict({
            'f': sp.Function('f', real=True, nonzero=True),
            'g': sp.Function('g', real=True, nonzero=True),
            'h': sp.Function('h', real=True, nonzero=True),
        })
        self.symbols = [
            'I', 'INT+', 'INT-', 'INT', 'FLOAT', '-', '.', '10^', 'Y', "Y'",
            "Y''"
        ]
        self.elements = [str(i) for i in range(abs(self.int_base))]

        # Vocabulary
        self.words = SPECIAL_WORDS + self.constants
        self.words += list(self.variables.keys())
        self.words += list(self.coefficients.keys())
        self.words += self.operators + self.symbols + self.elements
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # Indices
        self.n_words = len(self.words)
        self.eos_index = 0
        self.pad_index = 1

        # Dataset
        self.sym_math_dataset = {
            'data_train': [],
            'data_valid': [],
            'data_test': []
        }

    def data_config(self):
        return {
            'n_words': self.n_words,
            'eos_index': self.eos_index,
            'pad_index': self.pad_index,
            'id2word': self.id2word,
        }

    def prepare_data(self, *args, **kwargs) -> None:

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)

        for task in self.tasks:
            raw_path = RAW_DATA_DIRNAME / (task + ".tar.gz")
            processed_path = PROCESSED_DATA_DIRNAME / task

            if not os.path.exists(raw_path) or not os.path.exists(
                    processed_path):
                _download_and_process_sym_dataset(task, raw_path,
                                                  processed_path)

    def setup(self, stage=None):
        for split in ['train', 'valid', 'test']:
            temp_dataset = {'questions': [], 'answers': []}
            for task in self.tasks:
                path = PROCESSED_DATA_DIRNAME / task / (task + "." + split)
                with io.open(path, mode='r', encoding='utf-8') as f:
                    if self.max_elements == -1:
                        lines = [line.rstrip().split('|') for line in f]
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= self.max_elements:
                                break
                            lines.append(line.rstrip().split('|'))
                data = [xy.split('\t') for _, xy in lines]
                data = [xy for xy in data if len(xy) == 2]
                temp_dataset['questions'].extend([xy[0] for xy in data])
                temp_dataset['answers'].extend([xy[1] for xy in data])

            self.sym_math_dataset['data_' + split] = BaseDataset(
                temp_dataset['questions'], temp_dataset['answers'])

    def train_dataloader(self):
        return DataLoader(
            self.sym_math_dataset['data_train'],
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.sym_math_dataset['data_valid'],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.sym_math_dataset['data_test'],
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, batch):
        """
        Collate function for pytorch dataloader.
        """
        x, y = zip(*batch)
        nb_ops = [
            sum(int(word in OPERATORS) for word in seq.split()) for seq in x
        ]
        x = [self.encode_seq(seq) for seq in x]
        y = [self.encode_seq(seq) for seq in y]
        x, len_x = self.batch_sequences(x)
        y, len_y = self.batch_sequences(y)

        return (x, len_x), (y, len_y), torch.LongTensor(nb_ops)

    def encode_seq(self, seq):
        return torch.LongTensor(
            [self.word2id[w] for w in seq if w in self.word2id])

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([2 + len(s) for s in sequences])
        sent = torch.LongTensor(lengths.max().item(),
                                lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths


def _download_and_process_sym_dataset(task, raw_path, processed_path):
    _download_raw_dataset(task, raw_path)
    _process_raw_dataset(task, raw_path, processed_path)


def _download_raw_dataset(task, path):
    if path.exists():
        return path
    print(f"Downloading raw dataset to {path}...")
    url = SYM_URLS[task]
    download_url(url, path)
    return path


def _process_raw_dataset(task, raw_path, processed_path):
    if processed_path.exists():
        return processed_path
    processed_path.mkdir(parents=True, exist_ok=True)
    print(f"Processing raw dataset to {processed_path}...")
    os.system(f"tar -xvf {raw_path} -C {processed_path}")
