import json
import os
import glob
from collections import namedtuple
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *


with open('config.json') as file:
    cfg = json.load(file)

# DATA_FOLDER = cfg['DATA_FOLDER']
DATA_FOLDER = "data/rust/src/tests/ui/"
EXCLUDED_SUBDIRS = cfg['EXCLUDED_SUBDIRS']

Item = namedtuple("Item", "fname relpath")
"""Structure to store location of each file

fname (str) : filename without extension (filename.*)
relpath (str) : relative path to the file (tests/ui/{relpath}/filename.*)
"""


class UITestsDataset(Dataset):
    data_folder: str
    mode: str
    items: List[Item]

    def __init__(self, tests_ui_folder: str = DATA_FOLDER, mode: str = 'train') -> None:
        super(UITestsDataset).__init__()
        self.data_folder = tests_ui_folder
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

                print(len(files))
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

    def __getitem__(self, index: int) -> Tuple[str, str]:
        item = self.items[index]

        text = self.item2text(item)
        label = item.relpath.split(os.sep)[0]

        return (text, label)

        
train = UITestsDataset(mode='train')
infer = UITestsDataset(mode='infer')