import json
import os
import glob
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *

from src.params import *

Item = namedtuple("Item", "fname relpath")
"""Structure to store location of each file

fname (str) : filename without extension (filename.*)
relpath (str) : relative path to the file (tests/ui/{relpath}/filename.*)
"""

class UITestsDataset(Dataset):
    data_folder: str
    mode: str
    items: List[Item]

    def __init__(self, tests_ui_folder: str = PATH_TEST_UI, mode: str = 'train') -> None:
        super(UITestsDataset).__init__()
        self.data_folder = tests_ui_folder
        self.extensions = ['.rs', '.stderr', '.stdout']
        self.mode = mode
        self.items = self.get_items(tests_ui_folder, mode)

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

        text = []
        for fname in [f for f in glob.glob(f"{search}*") \
            if os.path.splitext(f)[1] in self.extensions]:

            with open(fname, 'r') as file:
                text.append(file.read())
        
        return '\n\n'.join(text)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[List[str], str]:
        item = self.items[index]
        text = self.item2text(item).split()
        label = item.relpath.split(os.sep)[0]
        return (text, label)

    @property
    def classes(self):
        return list(set([item.relpath.split(os.sep)[0] for item in self.items]))

    @property
    def vocab(self):
        vocab = set()
        for index in range(self.__len__()):
            words = self.__getitem__(index)[0]
            vocab.update(words)
        return list(vocab)


def tokens2features(tokens, generalized_tokens):
    """Prepare bag of features for each file
    Tokens, "generalized" tokens, and their bigrams and trigrams are used as features.
    """
    def twogram(l : List[str]):
        return [f"{l[i]} {l[i+1]}" for i in range(len(l) - 1)]
    
    def skipgram(l : List[str]):
        return [f"{l[i]} {l[i+2]}" for i in range(len(l) - 2)]
    
    def threegram(l : List[str]):
        return [f"{l[i]} {l[i+1]} {l[i+2]}" for i in range(len(l) - 2)]
    
    res = tokens + generalized_tokens
    res += twogram(tokens) + twogram(generalized_tokens)
    res += skipgram(tokens) + skipgram(generalized_tokens)
    res += threegram(tokens) + threegram(generalized_tokens)
    return res

class BaselineDataset(Dataset):
    tokens: UITestsDataset
    generalized_tokens: UITestsDataset
    mode: str

    def __init__(self, tokens_folder: str = PATH_PARSED_CLASSIFUI,
        generalized_tokens_folder: str = PATH_PARSED_CLASSIFUI_GENERALIZED, 
        mode: str = 'train') -> None:
        super(UITestsDataset).__init__()
        self.tokens = UITestsDataset(tokens_folder, mode)
        self.generalized_tokens = UITestsDataset(generalized_tokens_folder, mode)

    def __getitem__(self, index) -> Tuple[List[str], str]:
        tokens, label = self.tokens[index]
        generalized_tokens, _ = self.generalized_tokens[index]

        features = tokens2features(tokens, generalized_tokens)
        return (features, label)

    def __len__(self):
        return len(self.tokens)

    @property
    def vocab(self):
        vocab = set()
        for index in range(self.__len__()):
            words = self.__getitem__(index)[0]
            vocab.update(words)
        return list(vocab)