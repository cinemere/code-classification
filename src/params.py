# ----- These are default arguments of argparser in main.py ----- 
# setup experiment (argparser default variables)
EXPERIMENT_NAME = "TEST"
TRAIN_TEST_SPLIT = 0.7

METHOD = "baseline"
METHOD_CHOICES = ["baseline", "codeparrot"]

SEED = 42                                                   # used in traintestsplit in baseline/dataloader.py (modified and set in main)

MODE = "eval"                                               # in eval mode we perform train-test-split to measure accuracy
MODE_CHOICES = ["eval", "predict"]                          # in predict mode we use full dataset to train classifier

# liblinear params
LIBLINEAR_PARAMS = "-s 5"                                   # L1-regularization L2-loss SVC

# codeparrot params
CP_BATCH_SIZE = 4
CP_N_EPOCHS = 8
CP_MAX_SEQUENCE_LENGTH = 512
CP_LEARNING_RATE = 5e-5

# ----- These arguments are not passed to argparser im main.py -----
# paths
PATH_REPO = "/home/huawei123/kwx1991442/code-classification"

PATH_TEST_UI = "data/rust/tests/ui"

PATH_PARSED_CLASSIFUI             = "parser/parsed_data"
PATH_PARSED_CLASSIFUI_GENERALIZED = "parser/parsed_data_generalized"

PATH_CODEPARROT = "data/codeparrot-small"

PATH_SAVE_MODEL       = "saved_data/models"
PATH_SAVE_PREDICTIONS = "saved_data/predictions"
PATH_SAVE_METRICS     = "saved_data/metrics"
PATH_SAVE_LOGS        = "saved_data/logs"

# params of dataloader
EXCLUDED_SUBDIRS = [
    "auxiliary",
    "bad",
    "did_you_mean",
    "error-codes",
    "issues",
    "rfcs",
    "span",
    "suggestions"
]