"""Word2Vec embedding"""
import logging
import os
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Tokenizer
from typing import *

from src.params import *
from src.baseline.dataloader import UITestsDataset

logger = logging.getLogger(__name__)

class TokenizerSplitter(object):
    tokenizer: GPT2Tokenizer
    max_sequence_length : int = int(1e4)

    def __init__(self, path : str = PATH_CODEPARROT):
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        # default to left padding
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, text : str):
        return self.tokenizer.tokenize(text)

class LiblinearSplitter(object):
    def __call__(self, rawtext) -> List[str]:
        return rawtext.split()

class TrainW2VModel(object):
    """Train or load W2V model"""
    model: Word2Vec
    splitter: Union[LiblinearSplitter, TokenizerSplitter]
    method: str = "Word2Vec"

    def __init__(self, 
        train_set: UITestsDataset,
        tokens_source: str,
        min_count: int = W2V_MIN_COUNT, 
        vector_size: int = W2V_VECTOR_SIZE, 
        window: int = W2V_WINDOW, 
        epochs: int = W2V_EPOCHS,
        load_model: bool = False,
        load_model_path: str = W2V_MODEL_PATH
        ) -> None:
        self.load_model = load_model
        self.load_model_path = load_model_path
        
        self.tokens_source = tokens_source
        self.min_count = None
        self.vector_size = None
        self.window = None
        self.epochs = None
        
        logger.info(f"Setting up {self.method} model.")
        self.setup_splitter(tokens_source)
        self.setup_model(train_set, min_count, vector_size, window, epochs)
        logger.info(f"{self.get_modelname()} model is set up.")

    def parse_params_for_loading_model(self):
        min_count = int(self.load_model_path.split('/')[-1].split('_min_count=')[1].split('_')[0])
        vector_size = int(self.load_model_path.split('/')[-1].split('_vector_size=')[1].split('_')[0])
        window = int(self.load_model_path.split('/')[-1].split('_window=')[1].split('_')[0])
        epochs = int(self.load_model_path.split('/')[-1].split('_epochs=')[1].split('_')[0])

    def get_modelname(self):
        tokens_source = self.tokens_source
        min_count = self.min_count        
        vector_size = self.vector_size
        window = self.window
        epochs = self.epochs
        
        if self.load_model:
            loaded_exp_name = self.load_model_path.split('/')[-1].split('_min_count=')[0]
            return f"loaded={loaded_exp_name}_{min_count=}_{vector_size=}_{window=}_{epochs=}_tokens_source={tokens_source}"
        else:
            return f"{min_count=}_{vector_size=}_{window=}_{epochs=}_tokens_source={tokens_source}"

    def setup_splitter(self, tokens_source) -> None:
        if tokens_source == 'origin':
            self.splitter = TokenizerSplitter() 
        elif tokens_source == 'classifui':
            self.splitter = LiblinearSplitter()

    def setup_model(self, train_set, min_count, vector_size, window, epochs) -> None:
        """Train Word2Vec model
        
        sentences_train : List[List[str]]
        """
        sentences_train = [self.splitter(text) for text, _ in train_set]
        if not self.load_model:
            logger.info(f"Fitting {self.method} model")
            self.model = Word2Vec(sentences_train, 
                min_count=min_count,
                vector_size=vector_size, 
                window=window, 
                epochs=epochs)        
        else:
            logger.info(f"Loading {self.method} model")
            self.model = Word2Vec.load(self.load_model_path)
        
        self.min_count = self.model.min_count
        self.vector_size = self.model.vector_size
        self.window = self.model.window
        self.epochs = self.model.epochs

    def get_model(self) -> Tuple[Union[LiblinearSplitter, TokenizerSplitter], Word2Vec]:
        return self.splitter, self.model

    def save_model(self, experiment_name):
        if not self.load_model:
            logging.info(f"{self.method} model for {experiment_name} was loaded so it would not be saved.")
            return

        logger.info(f"Saving {self.method} model")
        model_folder = os.path.join(PATH_SAVE_MODEL, experiment_name)
        if not os.path.exists(model_folder): os.makedirs(model_folder)
        model_path = os.path.join(model_folder, f"model={self.method}")
        self.model.save(model_path)
        logging.info(f"Saved {self.method} model to : {model_path}")

class TrainD2VModel(TrainW2VModel):
    model : Doc2Vec
    splitter : Union[LiblinearSplitter, TokenizerSplitter]
    method: str = "Doc2Vec"

    def __init__(self, *args, **kwargs) -> None:
        kwargs['load_model_path'] = D2V_MODEL_PATH
        super().__init__(*args, **kwargs)

    def setup_model(self, train_set, min_count, vector_size, window, epochs) -> None:
        sentences_train = [self.splitter(text) for text, _ in train_set]
        tag_documents = [TaggedDocument(sentences_train[i], [i]) for i in range(len(sentences_train))]

        if not self.load_model:
            logger.info(f"Fitting {self.method} model")
            self.model=Doc2Vec( 
                min_count=min_count,
                vector_size=vector_size, 
                window=window,
                epochs=epochs,
                workers=6)        
            self.model.build_vocab(tag_documents)
            self.model.train(tag_documents, 
                total_examples=self.model.corpus_count,
                epochs=epochs)
        else:
            logger.info(f"Loading {self.method} model")
            self.model = Doc2Vec.load(self.load_model_path)

        self.min_count = self.model.min_count
        self.vector_size = self.model.vector_size
        self.window = self.model.window
        self.epochs = self.model.epochs

 
class W2VClassificationCollator(object):
    model : Word2Vec
    labelencoder : LabelEncoder
    splitter : Union[LiblinearSplitter, TokenizerSplitter]
    
    def __init__(self, 
        model : Word2Vec,
        splitter : Union[LiblinearSplitter, TokenizerSplitter], 
        classes : List[str]
        ) -> None:
        self.model = model
        self.splitter = splitter
        self.labelencoder = LabelEncoder().fit(classes)

    def encode_sentence(self, sentence):
        if len(sentence) == 0:
            return np.zeros(self.model.wv.vector_size)        
        else:
            return self.model.wv.get_mean_vector(sentence)

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        return self.labelencoder.transform(labels)

    def __call__(self, rawtexts_labels : List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
        rawtexts, labels = zip(*rawtexts_labels)
        vectors = [self.encode_sentence(self.splitter(text)) for text in rawtexts]
        encoded_labels = self.encode_labels(labels)
        return vectors, encoded_labels


class D2VClassificationCollator(W2VClassificationCollator):
    model : Doc2Vec
    labelencoder : LabelEncoder
    splitter : Union[LiblinearSplitter, TokenizerSplitter]

    def __init__(self, 
        model : Doc2Vec,
        splitter : Union[LiblinearSplitter, TokenizerSplitter], 
        classes : List[str]
        ) -> None:
        self.model = model
        self.splitter = splitter
        self.labelencoder = LabelEncoder().fit(classes)
    
    def encode_sentence(self, sentence):
        if len(sentence) == 0:
            return np.zeros(self.model.vector_size)        
        else:
            return self.model.infer_vector(sentence)