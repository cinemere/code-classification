# setup experiment (argparser default variables)
EXPERIMENT_NAME = "TEST"
METHOD = "baseline"
METHOD_CHOICES = ["baseline", "codeparrot"]

# paths
PATH_REPO = "/home/huawei123/kwx1991442/code-classification"
PATH_TEST_UI = "data/rust/tests/ui"
PATH_PARSED_CLASSIFUI = "parser/parsed_data"
PATH_PARSED_CLASSIFUI_GENERALIZED = "parser/parsed_data_generalized"
PATH_CODEPARROT = "data/codeparrot-small"

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

