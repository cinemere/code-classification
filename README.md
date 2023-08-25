# rust code classification

## setup

### Clone repo & install python packages
```
git clone https://github.com/cinemere/code-classification
cd code-classification
pip install -r requirements.txt
```
### Load data & pretrain
```
mkdir data
cd data
git clone https://github.com/rust-lang/rust
git lfs install
git clone https://huggingface.co/codeparrot/codeparrot-small-multi
cd ..
```
### Check default params
Go to src/params.py and fix the following path
```
PATH_REPO = "your/path/to/repo"
```
Also here you can view all params and change them.

### Run experiments
```
export PYTHONPATH=$PWD
python3 src/main.py --help
```
for GPU machine (to ignore some warnings):
```
export TF_CPP_MIN_LOG_LEVEL=2
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
[some-kind-of-siameze-metworks](https://github.com/IlyaGusev/tgcontest)

## results:

**baseline** validation
accuracy : mean=69.39 std=0.71\
mse : mean=4856.85 std=187.23\
sq_corr_coef : mean=0.48 std=0.02

**codeparrot** valudation accuaracy (on full dataset):\
0.681 maxlen=512 batch_size=4 epochs=5 lr=5e-5 (linear scheduler 8 epochs)

codeparrot valudation accuaracy (on small dataset, colab loading bug):\
0.735 maxlen=128 batch_size=4 epochs=4 lr=2e-5\
0.715 maxlen=512 batch_size=4 epochs=4 lr=2e-5\
0.761 maxlen=512 batch_size=4 epochs=5 lr=5e-5 (linear scheduler 8 epochs)

