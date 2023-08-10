import argparse
import logging
import time
import os
import numpy as np

from src.params import *
from baseline.dataloader import BaselineDataset

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--exp-name", type=str, default=EXPERIMENT_NAME,
        help="the name of this experiment")
    parser.add_argument("--method", type=str, choices=METHOD_CHOICES, default=METHOD,
        help="select method of classification")
    parser.add_argument("--seed", type=int, default=SEED, 
        help="seed of the experiment")
    parser.add_argument("--traintestsplit", type=float, default=0.7, 
        help="seed of the experiment")
    parser.add_argument("--mode", type=str, choices=MODE_CHOICES, default=MODE,
        help="eval mode is used to measure accuracy, predict mode to get prediction")
    parser.add_argument("--liblinear-params", type=str, default=LIBLINEAR_PARAMS,
        help="params of LIBLINEAR classifier")
    args = parser.parse_args()
    return args

def baseline(args, run_name):
    from liblinear.liblinearutil import (
        problem, parameter, train, predict, evaluations, save_model
    )

    # import data
    data = BaselineDataset()
    logging.info(f"{len(data.vocab)=}")
    logging.info(f"Dataset is setup!")

    if args.mode == "eval":
        logging.info(f"Baseline in {args.mode} mode.")
        
        # preprocessing    
        y_train, x_train, y_val, x_val = data.get_input_train_val(args.traintestsplit)
        logging.info(f"Data in splitted to train {len(y_train)=} and val {len(y_val)=}.")

        # training
        prob  = problem(y_train, x_train)
        param = parameter(args.liblinear_params)
        model = train(prob, param)
        logging.info(f"Model training is finished.")

        # evaluation
        predicted_labels, _, _ = predict(y_val, x_val, model)  # predicted_labels, accuracy, p_values
        logging.info(f"Validation labels are predicted.")

        (accuracy, mse, sq_corr_coef) = evaluations(y_val, predicted_labels)
        logging.info(f"{accuracy=} {mse=} {sq_corr_coef=}")
        
        # save model
        model_file_name = os.path.join(PATH_SAVE_MODEL, run_name)
        if not os.path.exists(PATH_SAVE_MODEL): os.makedirs(PATH_SAVE_MODEL)
        save_model(model_file_name, model)

        # save predictions
        predicted_folders = data.decode_predictions(predicted_labels)
        predictions_file_name = os.path.join(PATH_SAVE_PREDICTIONS, run_name)
        if not os.path.exists(PATH_SAVE_PREDICTIONS): os.makedirs(PATH_SAVE_PREDICTIONS)
        with open(f"{predictions_file_name}.txt", 'w') as f:
            for filename, folder in zip(data.filenames, predicted_folders):
                f.write(f"{filename} --> {folder}\n")
        
        # save metrics
        metrics_file_name = os.path.join(PATH_SAVE_METRICS, run_name)
        if not os.path.exists(PATH_SAVE_METRICS): os.makedirs(PATH_SAVE_METRICS)
        with open(f"{metrics_file_name}.txt", 'w') as f:
            f.write(f"{accuracy=} {mse=} {sq_corr_coef=}")

    elif args.mode == "predict":
        logging.info(f"Baseline in {args.mode} mode.")

        # preprocessing    
        y_train, x_train = data.get_input_data()
        logging.info(f"Input data is loaded {len(y_train)=}.")

        # training
        prob  = problem(y_train, x_train)
        param = parameter(args.liblinear_params)
        model = train(prob, param)
        logging.info(f"Model training is finished.")
        
        # prediction
        infer = BaselineDataset(mode='infer')
        _, x_infer = infer.get_input_data()

        predicted_labels, _, _ = predict([], x_infer, model)  # predicted_labels, accuracy, p_values
        
        # save model
        model_file_name = os.path.join(PATH_SAVE_MODEL, run_name)
        if not os.path.exists(PATH_SAVE_MODEL): os.makedirs(PATH_SAVE_MODEL)
        save_model(model_file_name, model)

        # save predictions
        predicted_folders = data.decode_predictions(predicted_labels)
        predictions_file_name = os.path.join(PATH_SAVE_PREDICTIONS, run_name)
        if not os.path.exists(PATH_SAVE_PREDICTIONS): os.makedirs(PATH_SAVE_PREDICTIONS)
        with open(f"{predictions_file_name}.txt", 'w') as f:
            for filename, folder in zip(infer.filenames, predicted_folders):
                f.write(f"{filename} --> {folder}\n")

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)

    run_name = f"{args.exp_name}_{args.method}_{args.seed}_{int(time.time())}"
    logging.info(f"{run_name=}")

    if args.method == "baseline":
        baseline(args, run_name)