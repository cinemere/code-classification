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