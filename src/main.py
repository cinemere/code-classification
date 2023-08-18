import argparse
import logging
from datetime import datetime
import numpy as np
import os

from src.params import *
from src.baseline.trainer import run_baseline
from src.codeparrot.trainer import run_codeparrot

def setup_logging(verbose, experiment_name):
    logs_folder = PATH_SAVE_LOGS
    if not os.path.exists(logs_folder): os.makedirs(logs_folder)

    logs_filename = os.path.join(logs_folder, f"logs_{experiment_name}.log")

    logging.basicConfig(
        filename=logs_filename, filemode="w",
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if verbose else logging.INFO
    )


def parse_args():
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

    # liblinear params
    liblinear_params = parser.add_argument_group("Liblinear parameters")
    liblinear_params.add_argument("--liblinear-params", type=str, default=LIBLINEAR_PARAMS,
        help="params of LIBLINEAR classifier")
    
    # codeparrot params
    codeparrot_params = parser.add_argument_group("Codeparrot parameters")
    codeparrot_params.add_argument("--batch-size", type=int, default=CP_BATCH_SIZE,
        help="batch size for codeparrot model")
    codeparrot_params.add_argument("--n-epochs", type=int, default=CP_N_EPOCHS,
        help="number of epochs for learning codeparrot model (and for scheduler)")
    codeparrot_params.add_argument("--max-seq-len", type=int, default=CP_MAX_SEQUENCE_LENGTH,
        help="maximum length of sequence of tokens")
    codeparrot_params.add_argument("-lr", "--learning-rate", type=int, default=CP_LEARNING_RATE,
        help="number of epochs for learning codeparrot model (and for scheduler)")
    codeparrot_params.add_argument('--device', type=str, default=DEVICE,
        help='device for codeparrit finetune')

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

    
    args = parser.parse_args()
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
    elif args.method == "codeparrot":
        run_codeparrot(args, run_name)