import argparse
import logging
import time

from src.params import *
from baseline.dataloader import BaselineDataset

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=EXPERIMENT_NAME,
        help="the name of this experiment")
    parser.add_argument("--method", type=str, choices=METHOD_CHOICES, default=METHOD,
        help="select method of classification")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_{args.method}_{int(time.time())}"
    logging.info(f"{run_name=}")

    data = BaselineDataset()
    logging.info(f"{len(data.vocab)=}")
    logging.info(f"Dataset is setup!")