import os
from tqdm import tqdm
from datetime import datetime
import logging
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import Generator
from transformers import (
    GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification,
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score, classification_report

from src.codeparrot.dataloader import ClassificationCollator
from src.baseline.dataloader import UITestsDataset
from src.params import (
    PATH_CODEPARROT,
    PATH_TEST_UI,
    PATH_SAVE_MODEL, PATH_SAVE_METRICS, PATH_SAVE_PREDICTIONS
)

logger = logging.getLogger(__name__)

class Trainer(object):
    model : GPT2ForSequenceClassification
    optimizer : AdamW
    loader_train : DataLoader
    loader_val : DataLoader
    loader_inf : DataLoader
    n_epochs : int
    def __init__(self,
        model,
        optimizer,
        device,
        loader_train, loader_val, loader_inf = None,
        n_epochs = 4,
        experiment_name = 'test',
        hparams = {}
        ) -> None:

        self.model = model
        self.loader_train, self.loader_val = loader_train, loader_val
        self.loader_inf = loader_inf

        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.scheduler = self.setup_scheduler()

        self.writer : SummaryWriter = self.setup_tensorboard(experiment_name, hparams)
        self.log_loss = {'train' : [], 'validation' : []}
        self.log_accu = {'train' : [], 'validation' : []}

        self.device = device

    def setup_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = self.n_epochs * len(self.loader_train))
        return scheduler

    def setup_tensorboard(self, experiment_name, hparams):        
        writer = SummaryWriter(f"runs/{experiment_name}")
        layout = {
            "metrics": {
                "loss": ["Multiline", ["loss/train", "loss/validation"]],
                "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
                "epoch_loss": ["Multiline", ["epoch_loss/train", "epoch_loss/validation"]],
                "epoch_accuracy": ["Multiline", ["epoch_accuracy/train", "epoch_accuracy/validation"]],
            },
        }
        writer.add_custom_scalars(layout)
        writer.add_hparams(hparams, {'metric' : 0})
        return writer

    def train_one_epoch(self, n_epoch):
        
        predicted_labels = []
        true_labels = []
        total_loss = 0

        self.model.train()
        for n_batch, (sequences, labels) in tqdm(enumerate(self.loader_train), total=len(self.loader_train)):

            self.model.zero_grad()

            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(**sequences, labels=labels)
            loss, logits = outputs[:2]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

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
            logger.info(f"train \tloss : {loss.item()}\taccu : {accu}\t{n_epoch * len(self.loader_train) + n_batch}")

        avg_epoch_loss = total_loss / len(self.loader_train)
        self.log(true_labels, predicted_labels, avg_epoch_loss, n_epoch, 'train')

    def evaluate(self, n_epoch):
        predicted_labels = []
        true_labels = []
        total_loss = 0

        self.model.eval()
        for n_batch, (sequences, labels) in tqdm(enumerate(self.loader_val), total=len(self.loader_val)):

            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(**sequences, labels=labels)
                loss, logits = outputs[:2]

            # log
            logits = logits.detach().cpu().numpy()
            predicted_labels += logits.argmax(axis=-1).flatten().tolist()
            true_labels += labels.detach().cpu().numpy().flatten().tolist()
            total_loss += loss.item()

            accu = accuracy_score(labels.detach().cpu().numpy().flatten().tolist(), logits.argmax(axis=-1).flatten().tolist())
            self.writer.add_scalar("loss/validation", loss.item(), n_epoch * len(self.loader_val) + n_batch)
            self.writer.add_scalar("accuracy/validation", accu, n_epoch * len(self.loader_val) + n_batch)
            logger.info(f"validation \tloss : {loss.item()}\taccu : {accu}\t{n_epoch * len(self.loader_val) + n_batch}")

        avg_epoch_loss = total_loss / len(self.loader_train)
        self.log(true_labels, predicted_labels, avg_epoch_loss, n_epoch, 'validation')

    def log(self, true_labels, predicted_labels, arg_epoch_loss, n_epoch, mode='train'):
        accu = accuracy_score(true_labels, predicted_labels)
        self.log_accu[mode].append(accu)
        self.log_loss[mode].append(arg_epoch_loss)
        logger.info(f"{mode}\tloss : {arg_epoch_loss}\taccu : {accu}")

        self.writer.add_scalar(f"epoch_loss/{mode}", arg_epoch_loss, n_epoch)
        self.writer.add_scalar(f"epoch_accuracy/{mode}", accu, n_epoch)

    def train(self):
        for n_epoch in tqdm(range(self.n_epochs)):
            self.train_one_epoch(n_epoch)
            self.evaluate(n_epoch)

def run_codeparrot(args, exp_name):
    
    if args.mode != 'eval':
        raise NotImplementedError()

    # Load data
    mode = 'train'
    tests_ui_folder=PATH_TEST_UI
    
    dataset = UITestsDataset(tests_ui_folder, mode)
    logger.info(f"UITestsDataset is initialized in mode {mode} from {tests_ui_folder}. {len(dataset)=}")

    # Split generator
    seed = args.seed
    lengths = [args.traintestsplit, 1-args.traintestsplit]

    splitgenerator = Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, 
        lengths=lengths, generator=splitgenerator)
    logger.info(f"Split generator initialized with {seed=}, {lengths=}.")

    # Train and validation dataloaders
    max_sequence_length = args.max_seq_len
    batch_size = args.batch_size

    collate_fn = ClassificationCollator(dataset.classes, max_sequence_length)
    loader_train = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn)
    loader_val = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)
    logger.info(f"DataLoaders are initialized with {batch_size=}, {max_sequence_length=}")
    logger.info(f"{len(dataset.classes)=}, {len(loader_train)=}, {len(loader_val)=}")

    # Load gpt2 model
    model_name_or_path = PATH_CODEPARROT
    # model_name_or_path = "codeparrot/codeparrot-small-multi"
    n_labels = len(dataset.classes)
    
    model_config = GPT2Config.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
    model = GPT2ForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, config=model_config)
    model.resize_token_embeddings(len(collate_fn.tokenizer))
    model.config.pad_token_id = model.config.eos_token_id

    # Setup trainer
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    codeparrot_params = f"{max_sequence_length=}_{batch_size=}_{learning_rate=}_{n_epochs=}"
    experiment_name = f"{exp_name}_{codeparrot_params}"
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model.to(device)
    optimizer = AdamW(model.parameters(),
                  lr = learning_rate, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )
    trainer = Trainer(model, optimizer, device, loader_train, loader_val, experiment_name=experiment_name, n_epochs=n_epochs, hparams=vars(args))

    trainer.train()

    # Save model
    save_path = PATH_SAVE_MODEL
    if not os.path.exists(save_path): os.makedirs(save_path)
    trainer.model.save_pretrained(os.path.join(save_path, f"model_{experiment_name}"))
    collate_fn.tokenizer.save_pretrained(os.path.join(save_path, f"tokenizer_{experiment_name}"))