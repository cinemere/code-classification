"""Word2Vec embedding on baseline tokens"""

from src.baseline.dataloader import UITestsDataset
from gensim.models import Word2Vec
from tqdm import tqdm
import numpy as np
import gensim
from gensim.models import Word2Vec

from src.params import *

class W2V_UITestsDataset(UITestsDataset):
    def __init__(self, 
        tests_ui_folder: str = PATH_PARSED_CLASSIFUI, mode: str = 'train', 
        traintestsplit : float = 0.7,
        make_encodings: bool = True
        ) -> None:
        super(W2V_UITestsDataset, self).__init__(tests_ui_folder, mode)
        self.make_traintestspit(traintestsplit)
        
        self.w2v_model = None
        self.train_enmbeddings()

        self.classes_encoder = None
        self.classes_decoder = None
        if make_encodings:
            self.make_classes_encodings()

    def make_traintestspit(self, traintestsplit : float = 0.7):
        self.train_idxs, self.val_idxs = [], []
        for index in tqdm(range(self.__len__())):
            if np.random.random() < traintestsplit:
                self.train_idxs.append(index)
            else:
                self.val_idxs.append(index)

    def make_classes_encodings(self):
        classes = self.classes
        self.classes_encoder = dict([(y, idx + 1) for idx, y in enumerate(classes)])
        self.classes_decoder = dict([(idx + 1, y) for idx, y in enumerate(classes)])
    
    def get_sentences(self, mode="full"):
        if mode == "full":
            return [self.__getitem__(index)[0] for index in tqdm(range(self.__len__()))]
        elif mode == "train":
            return [self.__getitem__(index)[0] for index in tqdm(self.train_idxs)]
        elif mode == "val":
            return [self.__getitem__(index)[0] for index in tqdm(self.val_idxs)]

    def train_enmbeddings(self, 
        min_count=5, vector_size=500, window=500, epochs=50
        ) -> None:

        sentences_train = self.get_sentences("train")
        sentences_val = self.get_sentences("val")
        
        self.w2v_model = gensim.models.Word2Vec(sentences_train, min_count = min_count,
                                    vector_size = vector_size, window = window, epochs=epochs)

    def simple_encode_sentence(self, sentence):
        if len(sentence) == 0:
            return np.zeros(self.w2v_model.wv.vector_size)        
        else:
            return self.w2v_model.wv.get_mean_vector(sentence)

    def encode_sentence(self, sentence, maxlen=2000):
        vector_size = self.w2v_model.wv.vector_size
        if len(sentence) == 0:
            return np.zeros(self.w2v_model.wv.vector_size)        
        else:
            return self.w2v_model.wv.get_mean_vector(sentence)


    def get_input_train_val(self):
        X_train, Y_train, X_val, Y_val = [], [], [], []

        def get_YX(mode='train'):
            if mode == 'train':
                idxs = self.train_idxs
            elif mode == 'val':
                idxs = self.val_idxs

            Y, X = [], []
            for index in tqdm(idxs):
                sample = self.__getitem__(index)
                vectorized_text = self.encode_sentence(sample[0])
                encoded_label = self.classes_encoder[sample[1]]
                X.append(vectorized_text)
                Y.append(encoded_label)
            return Y, X

        Y_train, X_train = get_YX('train')
        Y_val,   X_val   = get_YX('val')
        return Y_train, X_train, Y_val, X_val