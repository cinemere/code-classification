# ----- These are default arguments of argparser in main.py ----- 
# setup experiment (argparser default variables)
EXPERIMENT_NAME = "TEST"

METHOD = "baseline"
METHOD_CHOICES = ["baseline", "codeparrot"]

SEED = 42                                                   # used in traintestsplit in baseline/dataloader.py (modified and set in main)

MODE = "eval"                                               # in eval mode we perform train-test-split to measure accuracy
MODE_CHOICES = ["eval", "predict"]                          # in predict mode we use full dataset to train classifier

LIBLINEAR_PARAMS = "-s 5"                                   # L1-regularization L2-loss SVC

# ----- These arguments are not passed to argparser im main.py -----
# paths
PATH_REPO = "/home/huawei123/kwx1991442/code-classification"
PATH_TEST_UI = "data/rust/tests/ui"
PATH_PARSED_CLASSIFUI = "parser/parsed_data"
PATH_PARSED_CLASSIFUI_GENERALIZED = "parser/parsed_data_generalized"
PATH_CODEPARROT = "data/codeparrot-small"

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