# ----- These are default arguments of argparser in main.py ----- 
# setup experiment (argparser default variables)
EXPERIMENT_NAME = "TEST"
TRAIN_TEST_SPLIT = 0.7

METHOD = "baseline"
METHOD_CHOICES = ["baseline", "codeparrot", "word2vec"]

SEED = 42                                                   # used in traintestsplit in baseline/dataloader.py (modified and set in main)

MODE = "eval"                                               # in eval mode we perform train-test-split to measure accuracy
MODE_CHOICES = ["eval", "predict"]                          # in predict mode we use full dataset to train classifier

# dataset params
MIN_NUMBER_OF_FILES_IN_CLASS = 8

# liblinear params
LIBLINEAR_PARAMS = "-s 5"                                   # L1-regularization L2-loss SVC

# word2vec
TOKENS_SOURCE = 'classifui'
TOKENS_SOURCE_CHOICES = ['origin', 'classifui']
W2V_METHOD = "word2vec"
W2V_METHOD_CHOICES = ["word2vec", "doc2vec"]
W2C_CONCAT_METHOD = 'mean'
W2C_CONCAT_METHOD_CHOICES = ['mean', 'n-means']
W2V_MIN_COUNT = 5
W2V_VECTOR_SIZE = 500
W2V_WINDOW = 500
W2V_EPOCHS = 50
CLASSIFIER = 'bayes'
CLASSIFIER_CHOICES = ['logreg', 'forest', 'bayes', 'liblinear', 'all']

# codeparrot params
CP_BATCH_SIZE = 4
CP_N_EPOCHS = 8
CP_MAX_SEQUENCE_LENGTH = 512
CP_LEARNING_RATE = 5e-5
DEVICE = 'cuda'
SPLITTING = False

# ----- These arguments are not passed to argparser im main.py -----
# paths
PATH_REPO = "/home/huawei123/kwx1991442/code-classification"

PATH_TEST_UI = "data/rust/tests/ui"

PATH_PARSED_CLASSIFUI             = "parser/parsed_data"
PATH_PARSED_CLASSIFUI_GENERALIZED = "parser/parsed_data_generalized"

PATH_CODEPARROT = "data/codeparrot-small"
PATH_CODEGEN    = "data/codegen-350M-mono-rust/"

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

# trained models
W2V_MODEL_PATH = "/home/huawei123/kwx1991442/code-classification/saved_data/models/TEST_word2vec_42_24-Aug-20-53-19_min_count=5_vector_size=1500_window=1500_epochs=50_tokens_source='origin'"
# W2V_MODEL_PATH = '/home/huawei123/kwx1991442/code-classification/saved_data/models/word2vec_min-count=5_vector-size=500_window=500_epochs=50.model'
D2V_MODEL_PATH = ""