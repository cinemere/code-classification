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
├── config.json
├── data
│   ├── classifui
│   ├── codeparrot-small
│   ├── GPT2-News-Classifier
│   ├── monkey-rust
│   ├── rust
│   └── sandbox
├── main.py
├── notebooks
│   ├── baseline.ipynb
│   ├── Prepare_data.ipynb
│   └── test.ipynb
├── parser
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── model.json
│   ├── parsed_data                                                    # parsed data with in `classifui` style
│   ├── parsed_data_generalized                                        # parsed data with in `classifui` style (with generalization)
│   ├── src                                                            # parsing script
│   └── target
├── README.md
├── requirements.txt
└── src
    ├── dataloader.py
    └── __pycache__

15 directories, 11 files
```

## minor links
[python-tokenize-library](https://docs.python.org/3/library/tokenize.html#tokenize.generate_tokens)

