import logging
from typing import *

from torch import Generator
from torch.utils.data import random_split, DataLoader

from src.baseline.dataloader import UITestsDataset
from src.word2vec.dataloader import W2VClassificationCollator, TrainW2VModel
from src.params import PATH_TEST_UI, PATH_PARSED_CLASSIFUI

logger = logging.getLogger(__name__)


def run_word2vec(args, run_name):
    
    if args.mode != 'eval':
        raise NotImplementedError()

    # load data
    mode = 'train'
    tokens_source = args.tokens_source
    
    if tokens_source == 'origin':
        tests_ui_folder=PATH_TEST_UI
    elif tokens_source == 'classifui':
        tests_ui_folder=PATH_PARSED_CLASSIFUI

    dataset = UITestsDataset(tests_ui_folder, mode)
    logger.info(f"UITestsDataset is initialized in mode {mode} from {tests_ui_folder} "
                f" with {tokens_source=}. {len(dataset)=}")

    # split data
    seed = args.seed
    traintestsplit = args.traintestsplit
    lengths = [traintestsplit, 1-traintestsplit]

    splitgenerator = Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, 
        lengths=lengths, generator=splitgenerator)
    logger.info(f"Split generator initialized with {seed=}, {lengths=}.")

    # train w2v
    concat_method = args.w2v_concat_method #  TODO (NotImplemented)
    min_count = args.w2v_min_count
    vector_size = args.w2v_vector_size
    window = args.w2v_window
    epochs = args.w2v_epochs
    batch_size = 1 #args.batch_size

    splitter, w2v_model = TrainW2VModel(train_set, tokens_source, min_count, vector_size, window, epochs).get_model()
    collate_fn = W2VClassificationCollator(w2v_model, splitter, dataset.classes)
    loader_train = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn)
    loader_val   = DataLoader(  val_set, batch_size=batch_size, collate_fn=collate_fn)
    logger.info(f"DataLoaders are initialized with {batch_size=}")
    logger.info(f"{len(dataset.classes)=}, {len(loader_train)=}, {len(loader_val)=}")

    # prepare classifier, measure accuracy
    classifier = args.classifier
