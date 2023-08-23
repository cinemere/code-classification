# -*- coding: utf-8 -*-
"""Copy of finetune_codeparrot1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P4axtqrW50E79CSp4Wp_GE93cqWLYsdC
"""

# def visualize(self, rawtext: str, default_to_notebook: bool = True, mode: str = 'rawtext'):
#     r"""Visualize text from input

#     Problem: Not working with GPT2Tokenizer

#     Arguments:

#             rawtext (:obj: `str`):
#                 Raw text input.

#             default_to_notebook(:obj: `bool`, `optional`, defaults to :obj: `True`):
#                 output to notebook (or html)

#             mode (:obj: `str`, `optional`, defaults to :obj: `rawtext`):
#                 Options: `text`, `rawtext`

#     """
#     if mode == 'text':
#         text = self.encode_whitespaces(rawtext)
#         self.visualizer(text, default_to_notebook=default_to_notebook)
#     elif mode == 'rawtext':
#         self.visualizer(rawtext, default_to_notebook=default_to_notebook)

# Commented out IPython magic to ensure Python compatibility.
# %autosave 30
# %load_ext autoreload
# %autoreload 2

# cd ..

"""setup google-colab"""

from google.colab import drive
drive.mount('/content/drive/')

cd "/content/drive/MyDrive/Huawei/code-classification/code-classification"

ls

# !pip3 freeze

# rm -rf data/rust

# cd data

# !git clone https://github.com/rust-lang/rust

# !git lfs install
# !git clone https://huggingface.co/codeparrot/codeparrot-small-multi

# cd ..

import sys
sys.path.append('/content/drive/MyDrive/Huawei/code-classification/code-classification')

!pip3 install -r requirements.txt

"""load data"""

!nvidia-smi

from src.codeparrot.dataloader import UITestsLoader, ClassificationCollator
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch import Generator

dataset = UITestsLoader(mode='train', tests_ui_folder="/content/drive/MyDrive/Huawei/code-classification/code-classification/data/rust/tests/ui")
splitgenerator = Generator().manual_seed(42)
train_set, val_set = random_split(dataset, lengths=[0.7, 0.3], generator=splitgenerator)

# ls /content/drive/MyDrive/Huawei/code-classification/code-classification/data/rust/tests/ui | wc -l

collate_fn = ClassificationCollator(dataset.classes, max_sequence_length=512)
loader_train = DataLoader(train_set, batch_size=4, collate_fn=collate_fn)
loader_val = DataLoader(val_set, batch_size=4, collate_fn=collate_fn)
len(loader_train), len(loader_val)

"""load model"""

from src.params import PATH_CODEPARROT
from transformers import (
    GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification,
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup)

model_name_or_path = PATH_CODEPARROT
n_labels = len(dataset.classes)
n_labels

# # load auto model
# model_config = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path, num_labels=n_labels)
# model = AutoModelForSequenceClassification.from_pretrained(
#     pretrained_model_name_or_path=model_name_or_path, config=model_config, num_labels=n_labels)
# model.resize_token_embeddings(len(collate_fn.tokenizer))
# model.config.pad_token_id = model.config.eos_token_id

model_name_or_path = "codeparrot/codeparrot-small-multi"
# # load gpt2 model
model_config = GPT2Config.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path, config=model_config)
model.resize_token_embeddings(len(collate_fn.tokenizer))
model.config.pad_token_id = model.config.eos_token_id

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

"""setup"""

from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr = 5e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )

"""train"""

from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class Trainer(object):
    optimizer : AdamW
    loader_train : DataLoader
    loader_val : DataLoader
    loader_inf : DataLoader
    n_epochs : int
    def __init__(self,
        optimizer,
        loader_train, loader_val, loader_inf = None,
        n_epochs = 4,
        experiment_name = 'test'
        ) -> None:
        self.loader_train, self.loader_val = loader_train, loader_val
        self.loader_inf = loader_inf

        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.scheduler = self.setup_scheduler()

        self.writer : SummaryWriter = self.setup_tensorboard(experiment_name)
        self.log_loss = {'train' : [], 'validation' : []}
        self.log_accu = {'train' : [], 'validation' : []}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = self.n_epochs * len(self.loader_train))
        return scheduler

    def setup_tensorboard(self, experiment_name):
        now = datetime.now().strftime(f"%d-%b-%H-%M-%S")
        writer = SummaryWriter()
        layout = {
            "metrics": {
                "loss": ["Multiline", ["loss/train", "loss/validation"]],
                "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
                "epoch_loss": ["Multiline", ["epoch_loss/train", "epoch_loss/validation"]],
                "epoch_accuracy": ["Multiline", ["epoch_accuracy/train", "epoch_accuracy/validation"]],
            },
        }
        writer.add_custom_scalars(layout)
        return writer

    def train_one_epoch(self, n_epoch):
        predicted_labels = []
        true_labels = []
        total_loss = 0

        model.train()
        for n_batch, (sequences, labels) in tqdm(enumerate(self.loader_train), total=len(self.loader_train)):

            model.zero_grad()

            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            outputs = model(**sequences, labels=labels)
            loss, logits = outputs[:2]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            # log
            logits = logits.detach().cpu().numpy()
            predicted_labels += logits.argmax(axis=-1).flatten().tolist()
            true_labels += labels.detach().cpu().numpy().flatten().tolist()
            total_loss += loss.item()

            accu = accuracy_score(labels.detach().cpu().numpy().flatten().tolist(), logits.argmax(axis=-1).flatten().tolist())
            self.writer.add_scalar("loss/train", loss.item(), n_epoch * len(self.loader_train) + n_batch)
            self.writer.add_scalar("accuracy/train", accu, n_epoch * len(self.loader_train) + n_batch)
            print(f"train \tloss : {loss.item()}\taccu : {accu}\t{n_epoch * len(self.loader_train) + n_batch}")

        avg_epoch_loss = total_loss / len(self.loader_train)
        self.log(true_labels, predicted_labels, avg_epoch_loss, n_epoch, 'train')

    def evaluate(self, n_epoch):
        predicted_labels = []
        true_labels = []
        total_loss = 0

        model.eval()
        for n_batch, (sequences, labels) in tqdm(enumerate(self.loader_val), total=len(self.loader_val)):

            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(**sequences, labels=labels)
                loss, logits = outputs[:2]

            # log
            logits = logits.detach().cpu().numpy()
            predicted_labels += logits.argmax(axis=-1).flatten().tolist()
            true_labels += labels.detach().cpu().numpy().flatten().tolist()
            total_loss += loss.item()

            accu = accuracy_score(labels.detach().cpu().numpy().flatten().tolist(), logits.argmax(axis=-1).flatten().tolist())
            self.writer.add_scalar("loss/validation", loss.item(), n_epoch * len(self.loader_train) + n_batch)
            self.writer.add_scalar("accuracy/validation", accu, n_epoch * len(self.loader_train) + n_batch)

        avg_epoch_loss = total_loss / len(self.loader_train)
        self.log(true_labels, predicted_labels, avg_epoch_loss, n_epoch, 'validation')

    def log(self, true_labels, predicted_labels, arg_epoch_loss, n_epoch, mode='train'):
        accu = accuracy_score(true_labels, predicted_labels)
        self.log_accu[mode].append(accu)
        self.log_loss[mode].append(arg_epoch_loss)
        print(f"{mode}\tloss : {arg_epoch_loss}\taccu : {accu}")

        self.writer.add_scalar(f"epoch_loss/{mode}", arg_epoch_loss, n_epoch)
        self.writer.add_scalar(f"epoch_accuracy/{mode}", accu, n_epoch)

    def train(self):
        for n_epoch in tqdm(range(self.n_epochs)):
            self.train_one_epoch(n_epoch)
            self.evaluate(n_epoch)

trainer = Trainer(optimizer, loader_train, loader_val, experiment_name='batch-size-5_len-512', n_epochs=8)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir runs

trainer.train()

model.save_pretrained('/content/drive/MyDrive/Huawei/code-classification/code-classification/saved_data/models/batch-size-4_len-512_lr-5e-5')

collate_fn.tokenizer.save_pretrained('/content/drive/MyDrive/Huawei/code-classification/code-classification/saved_data/models/batch-size-4_len-512_lr-5e-5_tokenizer')

trainer.setup_scheduler()
for n_epoch in tqdm(range(trainer.n_epochs, trainer.n_epochs * 2)):
    trainer.train_one_epoch(n_epoch)
    trainer.evaluate(n_epoch)

del model

import torch
print(torch.cuda.memory_summary(device=None, abbreviated=False))

