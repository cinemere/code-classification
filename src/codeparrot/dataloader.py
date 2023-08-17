import json
import os
import glob
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import *

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer

from sklearn.preprocessing import LabelEncoder

from src.params import *

Item = namedtuple("Item", "fname relpath")
"""Structure to store location of each file

fname (str) : filename without extension (filename.*)
relpath (str) : relative path to the file (tests/ui/{relpath}/filename.*)
"""


class UITestDataset(Dataset):
    data_folder: str
    mode: str
    items: List[Item]
    
    # tokenizer
    tokenizer: GPT2Tokenizer #AutoTokenizer
    start_extra_id: int = 2
    stop_etra_id: int = 50
    max_sequence_length: int

    def __init__(self, tests_ui_folder: str = PATH_TEST_UI, mode: str = 'train', max_sequence_length : int = None) -> None:
        super(UITestDataset, self).__init__()
        self.data_folder = tests_ui_folder
        self.mode = mode
        self.items = self.get_items(tests_ui_folder, mode)
        self.labelencoder = LabelEncoder().fit(self.classes)
        self.setup_tokenizer()
        self.max_sequence_length =  self.tokenizer.model_max_length if max_sequence_length is None else max_sequence_length

    def setup_tokenizer(self, path : str = PATH_CODEPARROT):
        self.tokenizer = GPT2Tokenizer.from_pretrained(PATH_CODEPARROT)
        # default to left padding
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Add extra tokens to encode whitespaces
        extra_tokens = [f"<extratoken_{token_id}>" for token_id in range(self.start_extra_id, self.stop_etra_id)]
        self.tokenizer.add_tokens(new_tokens=extra_tokens, special_tokens=True)
        # Note. Don't forget to model.resize_token_embeddings(len(tokenizer)) if needed.

    def encode_whitespaces(self, text: str):
        """Encode whitespaces with extra tokens"""
        whitespace_tokens_ids = np.arange(self.start_extra_id, self.stop_etra_id)
        for token_id in whitespace_tokens_ids[::-1]:
            token = f"<extratoken_{token_id}>"
            text = text.replace(" " * token_id, token)
        return text

    def decode_whitespaces(self, text: str):
        """Decode the whitespace-encoded strings produced by encode_whitespaces"""
        whitespace_tokens_ids = np.arange(self.start_extra_id, self.stop_etra_id)
        for token_id in whitespace_tokens_ids:
            token = f"<extratoken_{token_id}>"
            text = text.replace(token, " " * token_id)
        return text

    def encode_sequence(self, rawtext: str) -> torch.Tensor:
        text = self.encode_whitespaces(rawtext)
        tokens = self.tokenizer.encode(
            text=text, 
            truncation=True,
            max_length=self.max_sequence_length,  # 1024
            return_tensors='pt'
        )
        return tokens

    def decode_sequence(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens)
        rawtext = self.decode_whitespaces(text)
        return rawtext

    def encode_batch(self, rawtexts : List[str]) -> torch.Tensor:
        texts = [self.encode_whitespaces(rawtext) for rawtext in rawtexts]
        tokens = self.tokenizer.encode(
            text=texts, 
            truncation=True,
            padding=True,  # for batch 
            max_length=self.max_sequence_length,  # 1024
            return_tensors='pt'
        )
        return tokens

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
        # rawtext = self.decode(self.encode(rawtext)[0]) #  TODO why this line?
        tokens = self.encode_sequence(rawtext)
        label = item.relpath.split(os.sep)[0]
        return (tokens, label)

    @property
    def classes(self):
        return list(set([item.relpath.split(os.sep)[0] for item in self.items]))


class UITestsLoader(Dataset):

    data_folder: str
    mode: str
    items: List[Item]
    
    def __init__(self, 
        tests_ui_folder: str = PATH_TEST_UI, 
        mode: str = 'train' 
        ) -> None:
        super(UITestsLoader).__init__()
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
        rawtext = self.item2text(item)
        label = item.relpath.split(os.sep)[0]
        return (rawtext, label)

    @property
    def classes(self):
        return list(set([item.relpath.split(os.sep)[0] for item in self.items]))


class ClassificationCollator(object):

    tokenizer: GPT2Tokenizer #  AutoTokenizer
    start_extra_id: int = 2
    stop_etra_id: int = 50
    max_sequence_length: int

    def __init__(self, classes : List[str], max_sequence_length : int = None) -> None:
        self.labelencoder = LabelEncoder().fit(classes)
        self.setup_tokenizer()
        self.max_sequence_length = self.tokenizer.model_max_length if max_sequence_length is None else max_sequence_length

    def setup_tokenizer(self, path : str = PATH_CODEPARROT):
        self.tokenizer = GPT2Tokenizer.from_pretrained(PATH_CODEPARROT)
        # default to left padding
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Add extra tokens to encode whitespaces
        extra_tokens = [f"<extratoken_{token_id}>" for token_id in range(self.start_extra_id, self.stop_etra_id)]
        self.tokenizer.add_tokens(new_tokens=extra_tokens, special_tokens=True)
        # Note. Don't forget to model.resize_token_embeddings(len(tokenizer)) if needed.

    def encode_whitespaces(self, text: str):
        """Encode whitespaces with extra tokens"""
        whitespace_tokens_ids = np.arange(self.start_extra_id, self.stop_etra_id)
        for token_id in whitespace_tokens_ids[::-1]:
            token = f"<extratoken_{token_id}>"
            text = text.replace(" " * token_id, token)
        return text

    def decode_whitespaces(self, text: str):
        """Decode the whitespace-encoded strings produced by encode_whitespaces"""
        whitespace_tokens_ids = np.arange(self.start_extra_id, self.stop_etra_id)
        for token_id in whitespace_tokens_ids:
            token = f"<extratoken_{token_id}>"
            text = text.replace(token, " " * token_id)
        return text

    def encode_sequence(self, rawtext: str) -> torch.Tensor:
        text = self.encode_whitespaces(rawtext)
        tokens = self.tokenizer.encode(
            text=text, 
            truncation=True,
            max_length=self.max_sequence_length,  # 1024
            return_tensors='pt'
        )
        return tokens

    def decode_sequence(self, tokens: List[int]) -> str:
        text = self.tokenizer.decode(tokens)
        rawtext = self.decode_whitespaces(text)
        return rawtext

    def encode_batch(self, rawtexts : List[str]) -> torch.Tensor:
        texts = [self.encode_whitespaces(rawtext) for rawtext in rawtexts]
        tokens = self.tokenizer(
            text=texts, 
            truncation=True,
            padding='max_length', #True,  # for batch. True -- max length of sequence in batch, max_length -- use maxlength 
            max_length=self.max_sequence_length,  # 1024
            return_tensors='pt'
        )
        return tokens

    def encode_labels(self, labels : List[str]) -> np.ndarray:
        # return [self.labelencoder[label] for label in labels]
        return self.labelencoder.transform(labels)

    def decode_labels(self, labels : np.ndarray) -> np.ndarray:
        # return [self.labelencoder.inverse_transform[label] for label in labels]
        return self.labelencoder.inverse_transform(labels)

    # def __getitem__(self, index: int) -> Tuple[List[int], str, str]:
    #     item = self.items[index]
    #     rawtext = self.item2text(item)
    #     # rawtext = self.decode(self.encode(rawtext)[0]) #  TODO why this line?
    #     tokens = self.encode_sequence(rawtext)
    #     label = torch.tensor(item.relpath.split(os.sep)[0])
    #     return (tokens, label)
    
    def __call__(self, rawtexts_labels : List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for torch DataLoader

        Returns:
                    tokens (Dict[str, torch.Tensor])
                        'input_ids'                     shape: [batch_size, maxlen] (maxlen=1024)
                        'attention_mask'                shape: [batch_size, maxlen] (maxlen=1024)

                    encodedlabels (torch.Tensor)        shape: [batch_size]
        """
        # print(len(rawtexts_labels), len(rawtexts_labels[0]), type(rawtexts_labels))
        rawtexts, labels = zip(*rawtexts_labels)
        tokens = self.encode_batch(rawtexts)
        encodedlabels = torch.from_numpy(self.encode_labels(labels))
        return tokens, encodedlabels