import argparse
import logging
from datetime import datetime
import numpy as np
import os

from src.params import *
from src.baseline.trainer import run_baseline
from src.codeparrot.trainer import run_codeparrot
from src.word2vec.trainer import run_word2vec

def setup_logging(verbose, experiment_name):
    logs_folder = PATH_SAVE_LOGS
    if not os.path.exists(logs_folder): os.makedirs(logs_folder)

    logs_filename = os.path.join(logs_folder, f"logs_{experiment_name}.log")

    logging.basicConfig(
        filename=logs_filename, filemode="w",
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO
    )

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(notebook=False, request=""):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # common params
    common_params = parser.add_argument_group("Common parameters")
    common_params.add_argument("--exp-name", type=str, default=EXPERIMENT_NAME,
        help="the name of this experiment")
    common_params.add_argument("--method", type=str, choices=METHOD_CHOICES, default=METHOD,
        help="select method of classification")
    common_params.add_argument("--seed", type=int, default=SEED, 
        help="seed of the experiment")
    common_params.add_argument("--traintestsplit", type=float, default=TRAIN_TEST_SPLIT, 
        help="seed of the experiment")
    common_params.add_argument("--mode", type=str, choices=MODE_CHOICES, default=MODE,
        help="eval mode is used to measure accuracy, predict mode to get prediction")
    common_params.add_argument("--load", action="store_true", 
        help="use already trained loaded model")

    # dataset params
    dataset_params = parser.add_argument_group("Dataset parameters")
    dataset_params.add_argument("-min", "--min-number-of-files-in-class", type=int,
        default=MIN_NUMBER_OF_FILES_IN_CLASS, help="remove data with low poplutaion classes")
    dataset_params.add_argument("--debug", action="store_true",
        help="enable debug mode (shortened version of dataset)")

    # liblinear params
    liblinear_params = parser.add_argument_group("Liblinear parameters")
    liblinear_params.add_argument("--liblinear-params", type=str, default=LIBLINEAR_PARAMS,
        help="params of LIBLINEAR classifier")

    # word2vec params
    word2vec_params = parser.add_argument_group("Word2vec parameters")
    word2vec_params.add_argument("--tokens-source", type=str, default=TOKENS_SOURCE,
        choices=TOKENS_SOURCE_CHOICES, help="the source of tokens for word2vec model")
    word2vec_params.add_argument("--w2v-method", type=str, default=W2V_METHOD,
        choices=W2V_METHOD_CHOICES, help="specify the model for training embeddings")
    word2vec_params.add_argument("--classifier", type=str, default=CLASSIFIER, 
        choices=CLASSIFIER_CHOICES, help="select classification method")
    word2vec_params.add_argument("--w2v-concat-method", type=str, default=W2C_CONCAT_METHOD,
        choices=W2C_CONCAT_METHOD_CHOICES, help="select way to concat w2v vectors to encode documents")
    word2vec_params.add_argument("--w2v-min-count", type=int, default=W2V_MIN_COUNT,
        help="Ignores all words with total frequency lower than this.")
    word2vec_params.add_argument("--w2v-vector-size", type=int, default=W2V_VECTOR_SIZE,
        help="Dimensionality of the word vectors.")
    word2vec_params.add_argument("--w2v-window", type=int, default=W2V_WINDOW,
        help="Maximum distance between the current and predicted word within a sentence.")
    word2vec_params.add_argument("--w2v-epochs", type=int, default=W2V_EPOCHS,
        help="Number of epochs to train w2v vectors.")
    
    # codeparrot params
    codeparrot_params = parser.add_argument_group("Codeparrot parameters")
    codeparrot_params.add_argument("--batch-size", type=int, default=CP_BATCH_SIZE,
        help="batch size for codeparrot model")
    codeparrot_params.add_argument("--n-epochs", type=int, default=CP_N_EPOCHS,
        help="number of epochs for learning codeparrot model (and for scheduler)")
    codeparrot_params.add_argument("--max-seq-len", type=int, default=CP_MAX_SEQUENCE_LENGTH,
        help="maximum length of sequence of tokens")
    codeparrot_params.add_argument("-lr", "--learning-rate", type=float, default=CP_LEARNING_RATE,
        help="number of epochs for learning codeparrot model (and for scheduler)")
    codeparrot_params.add_argument('--device', type=str, default=DEVICE,
        help='device for codeparrot finetune')
    codeparrot_params.add_argument('--splitting', type=str2bool, default=SPLITTING,
        help="enabled / disabled splitting of document (to encounter document ending in long documents)")

    # saving params
    saving_params = parser.add_argument_group("Saving params")
    saving_params.add_argument('--save-model', action='store_true', 
        help="use this flag to save the model")
    saving_params.add_argument('--save-predictions', action='store_true', 
        help="use this flag to save predictions")
    saving_params.add_argument('--save-metrics', action='store_true', 
        help="use this flag to save metrics")
    saving_params.add_argument('--save-all', action='store_true', 
        help="use this flag to save metrics, prediction and model")
    saving_params.add_argument("-v", "--verbose", action="store_true",
        help="logging in debug mode")

    args = parser.parse_args(request) if notebook else parser.parse_args()

    if args.save_all:
        args.save_model, args.save_predictions, args.save_metrics = True, True, True
    return args

if __name__ == "__main__":
    args = parse_args()
    
    now = datetime.now().strftime(f"%d-%b-%H-%M-%S")
    run_name = f"{args.exp_name}_{args.method}_{args.seed}_{now}"
    setup_logging(args.verbose, run_name)
    logging.info(f"{run_name=}")

    if args.method == "baseline":
        np.random.seed(args.seed)
        run_baseline(args, run_name)

    elif args.method == "word2vec":
        run_word2vec(args, run_name)
    
    elif args.method == "codeparrot":
        run_codeparrot(args, run_name)