import json
import os
import glob
from collections import namedtuple
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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
        rawtext = self.item2text(item)
        label = item.relpath.split(os.sep)[0]
        return (rawtext, label)

    @property
    def classes(self):
        return sorted(list(set([item.relpath.split(os.sep)[0] for item in self.items])))

    @property
    def vocab(self):
        vocab = set()
        for index in range(self.__len__()):
            words = self.__getitem__(index)[0].split()
            vocab.update(words)
        return sorted(list(vocab))

    @property
    def filenames(self):
        return [item.fname for item in self.items]

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
        mode: str = 'train', make_encodings: bool = True) -> None:
        super(UITestsDataset).__init__()
        self.tokens = UITestsDataset(tokens_folder, mode)
        self.generalized_tokens = UITestsDataset(generalized_tokens_folder, mode)
        self.features_encoder = None
        self.classes_encoder = None
        self.classes_decoder = None
        self.val_idxs = [idx for idx in range(len(self.tokens))]
        if make_encodings:
            self.make_encodings()

    def __getitem__(self, index) -> Tuple[List[str], str]:
        tokens, label = self.tokens[index]
        generalized_tokens, _ = self.generalized_tokens[index]

        features = tokens2features(tokens.split(), generalized_tokens.split())
        return (features, label)

    def __len__(self):
        return len(self.tokens)

    @property
    def classes(self):
        return self.tokens.classes

    @property
    def vocab(self):
        vocab = set()
        for index in tqdm(range(self.__len__())):
            words = self.__getitem__(index)[0]
            vocab.update(words)
        return sorted(list(vocab))
    
    @property
    def filenames(self):
        filenames = self.tokens.filenames
        return [filenames[idx] for idx in self.val_idxs]

    def make_encodings(self):
        """Prepare label encoder for features and classes"""
        self.features_encoder = dict([(y, idx + 1) for idx, y in enumerate(self.vocab)])
        classes = self.classes
        self.classes_encoder = dict([(y, idx + 1) for idx, y in enumerate(classes)])
        self.classes_decoder = dict([(idx + 1, y) for idx, y in enumerate(classes)])

    def get_input_data(self) -> Tuple[List[int], List[Dict[int, int]]]:
        """Form data to the format needed for liblinear classifier

        from liblinear docs:
        y: a Python list/tuple/ndarray of l labels (type must be int/double).
        x: 1. a list/tuple of l training instances. Feature vector of
           each training instance is a list/tuple or dictionary.
           2. an l * n numpy ndarray or scipy spmatrix (n: number of features).
        
        Output:
            Y (List[int]) : categorical labels
            X (List[Dict[int, int]]) : dictionary of training instances 
            (example: for feature vector [0, 1, 0, 1] X would be {1 : 1, 3 : 1})
        """
        X, Y = [], []
        for index in tqdm(range(self.__len__())):
            sample = self.__getitem__(index)
            encoded_features = {self.features_encoder[t] : 1. for t in sample[0]}
            encoded_label = self.classes_encoder[sample[1]]
            X.append(encoded_features)
            Y.append(encoded_label)
        return Y, X

    def get_input_train_val(self, traintestsplit : float = 0.7) -> Tuple[List[int], List[Dict[int, int]]]:
        """Form data to the format needed for liblinear classifier

        from liblinear docs:
        y: a Python list/tuple/ndarray of l labels (type must be int/double).
        x: 1. a list/tuple of l training instances. Feature vector of
           each training instance is a list/tuple or dictionary.
           2. an l * n numpy ndarray or scipy spmatrix (n: number of features).
        
        Output:
            Y_train (List[int]) : categorical labels
            X_train (List[Dict[int, int]]) : dictionary of training instances 
            (example: for feature vector [0, 1, 0, 1] X would be {1 : 1, 3 : 1})
            Y_val (List[int]) : categorical labels
            X_val (List[Dict[int, int]]) : dictionary of training instances 
            (example: for feature vector [0, 1, 0, 1] X would be {1 : 1, 3 : 1})
        """
        X_train, Y_train, X_val, Y_val, self.val_idxs = [], [], [], [], []
        for index in tqdm(range(self.__len__())):
            sample = self.__getitem__(index)
            encoded_features = {self.features_encoder[t] : 1. for t in sample[0]}
            encoded_label = self.classes_encoder[sample[1]]

            if np.random.random() < traintestsplit:
                X_train.append(encoded_features)
                Y_train.append(encoded_label)
            else:
                X_val.append(encoded_features)
                Y_val.append(encoded_label)
                self.val_idxs.append(index)
        return Y_train, X_train, Y_val, X_val

    def decode_predictions(self, predicted_labels: List[int]) -> List[str]:
        """Decode predicted liblinear classifier labels to str format (names of folders)"""
        return [self.classes_decoder[p_label] for p_label in predicted_labels]