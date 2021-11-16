from pathlib import Path
import pytorch_lightning as pl
import os
from deep_math.data.util import download_url
from deep_math.constants import SYM_URLS
import io
from torch.utils.data import DataLoader
from deep_math.data.base_dataset import BaseDataset

# Define file paths
DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"
RAW_DATA_DIRNAME = DATA_DIR / "raw_sym_datasets"
PROCESSED_DATA_DIRNAME = DATA_DIR / "symbolic_datasets"


class SymDataModule(pl.LightningDataModule):

    def __init__(self, tasks):
        super().__init__()
        self.batch_size = 32
        self.num_workers = 4  # Count of subprocesses to use for data loading
        self.tasks = tasks
        self.max_elements = 1

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

        self.sym_math_dataset = {
            'data_train': [],
            'data_valid': [],
            'data_test': []
        }

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
                print(f"Loaded {len(data)} {split} examples for {task}")

            self.sym_math_dataset['data_train'] = BaseDataset(
                temp_dataset['questions'], temp_dataset['answers'])

        x, y = next(iter(self.train_dataloader()))

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
            self.sym_math_dataset['data_val'],
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

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        print(elements)
        return elements


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