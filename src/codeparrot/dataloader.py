import json
import os
import glob
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *

from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers.tools import EncodingVisualizer

from sklearn.preprocessing import LabelEncoder

from src.params import *

Item = namedtuple("Item", "fname relpath")
"""Structure to store location of each file

fname (str) : filename without extension (filename.*)
relpath (str) : relative path to the file (tests/ui/{relpath}/filename.*)
"""


def decode_whitespaces(text: str, start_extra_id: int, max_len: int):
    """Decode the whitespace-encoded strings produced by encode_whitespaces"""
    for l in range(2, max_len + 1):
        token_id = start_extra_id - 2 + l
        token = f"<extratoken_{token_id}>"
        text = text.replace(token, ' ' * l)
    return text


class UITestsDataset(Dataset):
    data_folder: str
    mode: str
    items: List[Item]
    tokenizer: AutoTokenizer
    start_extra_id: int = 0
    max_len: int = 100

    def __init__(self, tests_ui_folder: str = PATH_TEST_UI, mode: str = 'train') -> None:
        super(UITestsDataset).__init__()
        self.data_folder = tests_ui_folder
        self.mode = mode
        self.items = self.get_items(tests_ui_folder, mode)

        self.tokenizer = AutoTokenizer.from_pretrained(PATH_CODEPARROT)
        self.visualizer = EncodingVisualizer(self.tokenizer._tokenizer)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.labelencoder = LabelEncoder().fit(self.classes)

    def get_items(self, test_ui_folder: str, mode: str) -> List[Item]:
        items = []

        if mode == 'train':
            for path, _, files in os.walk(test_ui_folder):

                # Check that file is placed in a subdirectory
                if len(path) == len(test_ui_folder):
                    continue

                relpath = os.path.relpath(path, start=test_ui_folder)
                
                # Check that subdirectory should not be excluded
                if len(set(relpath.split(os.sep)) & set(EXCLUDED_SUBDIRS)) > 0:
                    continue

                for f in files:
                    fname, fext = os.path.splitext(f)

                    # Omit non .rs files
                    if fext != ".rs":
                        continue

                    new_item = Item(fname, relpath)
                    items.append(new_item)

        elif mode == 'infer':
            for path, _, files in os.walk(test_ui_folder):

                # Check that file is NOT placed in a subdirectory
                if len(path) != len(test_ui_folder):
                    continue

                for f in files:
                    fname, fext = os.path.splitext(f)

                    # Omit non .rs files
                    if fext != ".rs":
                        continue

                    new_item = Item(fname, '')
                    items.append(new_item)

                # Skip all other paths as they are in a subdirectories
                break
        else:
            return ValueError("Unknown `mode` passed to `get_items()` function. " 
            "Possible variants: `train` or `infer`. ")

        return items

    def item2text(self, item: Item) -> str:
        """Load and concatenate files for item
        
        {item.fname}.[rs,stderr,stdout]
        """
        search = os.path.join(self.data_folder, item.relpath, item.fname)
        extensions = ['.rs', '.stderr', '.stdout']

        text = []
        for fname in [f for f in glob.glob(f"{search}*") \
            if os.path.splitext(f)[1] in extensions]:

            with open(fname, 'r') as file:
                text.append(file.read())
        
        return '\n\n'.join(text)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[List[int], str, str]:
        item = self.items[index]
        rawtext = self.item2text(item)
        rawtext = self.decode(self.encode(rawtext)[0])
        tokens = self.encode(rawtext)
        label = item.relpath.split(os.sep)[0]
        return (tokens, rawtext, label)

    def encode(self, rawtext: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(
            self.encode_whitespaces(rawtext, self.start_extra_id, self.max_len), 
            truncation=True,
            padding=True,  # for batch 
            max_length=1024,
            return_tensors='pt'
        )
        return tokens

    def decode(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens, )
        return decode_whitespaces(text, self.start_extra_id, self.max_len)

    @property
    def classes(self):
        return list(set([item.relpath.split(os.sep)[0] for item in self.items]))

    def visualize(self, rawtext: str, default_to_notebook: bool = True):
        text = self.encode_whitespaces(rawtext, self.start_extra_id, self.max_len)
        self.visualizer(rawtext, default_to_notebook=default_to_notebook)

    def encode_whitespaces(self, text: str, start_extra_id: int, max_len: int):
        """Encode whitespaces with extra tokens"""
        added_tokens = set()
        for i in np.arange(max_len, 1, -1):
            token_id = start_extra_id + i - 2
            token = f"<extratoken_{token_id}>"
            text = text.replace(" " * i, token)
            added_tokens.add(token)

        new_tokens = added_tokens - set(self.tokenizer.vocab.keys())
        self.tokenizer.add_tokens(list(new_tokens))
        return text