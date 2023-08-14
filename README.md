# rust code classification

## setup
```
git clone https://github.com/cinemere/code-classification
cd code-classification
pip install -r requirements.txt
export PYTHONPATH=$PWD
python3 src/main.py --help
```

## repo structure
```
.
├── data
│   ├── classifui
│   ├── codeparrot-small
│   ├── GPT2-News-Classifier
│   ├── monkey-rust
│   ├── rust
│   └── sandbox
├── notebooks
│   ├── baseline_sandbox.ipynb
│   ├── Prepare_data.ipynb
│   └── test.ipynb
├── parser
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── model.json
│   ├── parsed_data                                                    # parsed data with in `classifui` style [gitignore]
│   ├── parsed_data_generalized                                        # parsed data with in `classifui` style (with generalization) [gitignore]
│   ├── src                                                            # parsing script
│   └── target
├── README.md
├── requirements.txt
├── saved_data                                                         # [gitignore]
│   ├── metrics
│   ├── models
│   └── predictions
└── src
    ├── baseline
    ├── codeparrot
    ├── main.py
    ├── params.py
    └── __pycache__

21 directories, 10 files
```

## minor links
[python-tokenize-library](https://docs.python.org/3/library/tokenize.html#tokenize.generate_tokens)

## results:

baseline accuracy : mean=69.39 std=0.71
baseline mse : mean=4856.85 std=187.23
baseline sq_corr_coef : mean=0.48 std=0.02