import logging
from typing import *
import pickle
import os

import numpy as np
from torch import Generator
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from liblinear.liblinearutil import (
    problem, parameter, train, predict, evaluations, save_model
)
from src.utils import CustomSummaryWriter
from src.baseline.dataloader import UITestsDataset, RemovedBadClasses
from src.word2vec.dataloader import (W2VClassificationCollator, TrainW2VModel,
    D2VClassificationCollator, TrainD2VModel)
from src.params import (PATH_TEST_UI, PATH_PARSED_CLASSIFUI,
    PATH_SAVE_METRICS, PATH_SAVE_MODEL, PATH_SAVE_PREDICTIONS)

logger = logging.getLogger(__name__)

classifiers = {
    'logreg' : LogisticRegression(),
    'forest' : RandomForestClassifier(max_depth=5, random_state=0),
    'bayes_bernoulli' : BernoulliNB(),
    'bayes_gaussian' : GaussianNB(),
    'liblinear' : LogisticRegression(solver='liblinear'),
}
# prob  = problem(Y_train, X_train)
# param = parameter('-s 5') #  '-s 5'
# model = train(prob, param)
# predicted_labels, _, _ = predict(Y_val, X_val, model)

search_params = {
    'logreg' : {'penalty' : ('l1', 'l2', 'elasticnet', None), 'solver' : ('newton-cg',)},
    'forest' : {'max_depth' : (5, 10, 50), 'criterion' : ('gini', 'entropy', 'log_loss')},
    'bayes_bernoulli' : {'alpha' : (1., 2., 0.5), 'binarize' : (0., 0.25, 0.5)},
    'bayes_gaussian' : {'var_smoothing' : (1e-9, 1e-8)},
    'liblinear' : {'penalty' : ('l1', 'l2', None)}
}

def evaluate(y_val, y_pred, labelencoder) -> Tuple[Dict, Dict, Dict]:
    labels = labelencoder.classes_
    encoded_labels = np.arange(len(labels))
    m_vals, m_arrs, m_count = {}, {}, {}

    # averaged metrics
    m_vals['accuracy'] = accuracy_score(y_val, y_pred)
    m_vals['micro_precision'], m_vals['micro_recall'], m_vals['micro_fbeta_score'], _ = precision_recall_fscore_support(
        y_val, y_pred, average='micro', zero_division=1., labels=encoded_labels)
    m_vals['macro_precision'], m_vals['macro_recall'], m_vals['macro_fbeta_score'], _ = precision_recall_fscore_support(
        y_val, y_pred, average='micro', zero_division=0., labels=encoded_labels)
    
    # by-class metrics
    m_arrs['precision'], m_arrs['recall'], m_arrs['fbeta_score'], _ = precision_recall_fscore_support(
        y_val, y_pred, average=None, zero_division=1., labels=encoded_labels)

    # counts
    m_count['counts_y_val_x'], m_count['counts_y_val'] = np.unique(y_val,  return_counts=True)
    m_count['counts_y_pred_x'], m_count['counts_y_pred'] = np.unique(y_pred, return_counts=True)

    return m_vals, m_arrs, m_count

class Trainer(object):
    def __init__(self, classifier_name, logtb=True, experiment_name='', labelencoder=None, 
        save_model=True, save_predictions=True, save_metrics=True
        ) -> None:
        self.clf = GridSearchCV(classifiers[classifier_name], search_params[classifier_name])
        self.logtb = logtb
        self.classifier_name = classifier_name
        self.experiment_name = experiment_name
        self.labelencoder = labelencoder
        self.writer : CustomSummaryWriter = None
        self._save_model = save_model
        self._save_metrics = save_metrics
        self._save_predictions = save_predictions

    def setup_tensorboard(self):
        log_path = f'runs/{self.experiment_name}_clf-{self.classifier_name}'
        self.writer = CustomSummaryWriter(log_path)

    def log_tensorboard(self, metrics : Tuple[Dict, Dict, Dict]):
        if not self.logtb:
            return
        
        m_vals, m_arrs, m_count = metrics
        hparams = {key : str(value) for key, value in self.clf.best_params_.items()}
        self.writer.add_hparams(
            dict(hparams, clf=self.classifier_name),
            {f"hparam/{key}" : value for key, value in m_vals.items()}
        )
        for key, arr in m_arrs.items():
            for i, value in enumerate(arr):
                self.writer.add_scalar(f'each_class/{key}', value, i)

        for i, value in enumerate(m_count['counts_y_val']):
            self.writer.add_scalar(f'counts/y_val', value, i) 
        for i in range(len(m_count['counts_y_pred'])):
            self.writer.add_scalar(f'counts/y_pred', m_count['counts_y_pred'][i], m_count['counts_y_pred_x'][i]) 

        self.writer.add_text('hparams', '\n'.join([f"{key} : {value}" for key, value in hparams.items()]), 1)
        self.writer.add_text('metrics', '\n'.join([f"{key} : {value}" for key, value in m_vals.items()]), 1)
        self.writer.add_text('classifier', self.classifier_name, 1)
        self.writer.add_text('experiment', self.experiment_name, 1)
        self.writer.close()

    def fit(self, X_train, Y_train):
        logger.info(f"Fitting {self.classifier_name} classifier ({self.experiment_name})")
        self.clf.fit(X_train, Y_train)
        if self.logtb: self.setup_tensorboard()

    def eval(self, X_val, Y_val):
        logger.info(f"Evaluating {self.classifier_name} classifier ({self.experiment_name})")
        Y_pred = self.clf.best_estimator_.predict(X_val)
        metrics = evaluate(Y_pred, Y_val, self.labelencoder)
        
        self.log_tensorboard(metrics)
        self.save_metrics(metrics)
        self.save_model()
        self.save_predictions(Y_pred, Y_val)

    def save_metrics(self, metrics):
        if not self._save_metrics:
            return
        logger.info(f"Saving metrics for {self.classifier_name} classifier ({self.experiment_name})")

        m_vals, m_arrs, m_count = metrics

        metrics_folder = os.path.join(PATH_SAVE_METRICS, self.experiment_name)
        if not os.path.exists(metrics_folder): os.makedirs(metrics_folder)
        hparams = '_'.join([f"{key}-{str(value)}" for key, value in self.clf.best_params_.items()])
        metrics_filename = os.path.join(metrics_folder, f"{self.classifier_name}_{hparams}.txt")
        with open(metrics_filename, 'w') as f:

            f.write(f"AVERAGE METRICS:\n")
            for key, value in m_vals.items():
                f.write(f"{key:30}={value:10.8}\n")

            f.write(f"\nBY-CLASS METRICS:\n")
            for key, values in m_arrs.items():
                f.write(f"{key}:\n")
                for i, value in enumerate(values):
                    classname = f"({self.labelencoder.inverse_transform([i])[0]})"
                    f.write(f"{i}\t{classname:30}\t{value:10.4}\n")
            
            f.write(f"\nCOUNTS:\n")
            for i, (count_x, count_y) in enumerate(zip(m_count['counts_y_pred_x'], m_count['counts_y_pred'])):
                classname = f"({self.labelencoder.inverse_transform([count_x])[0]})"
                if count_x in m_count['counts_y_val_x']:
                    count_v = m_count['counts_y_val'][m_count['counts_y_val_x'] == count_x][0]
                else:
                    count_v = 0
                f.write(f"{count_x:3}\t{classname:30}\t{count_y:3} (out of {count_v:6})\n")

    def save_predictions(self, predictions, ground_true):
        if not self._save_predictions:
            return
        logger.info(f"Saving predictions for {self.classifier_name} classifier ({self.experiment_name})")

        predictions_folder = os.path.join(PATH_SAVE_PREDICTIONS, self.experiment_name)
        if not os.path.exists(predictions_folder): os.makedirs(predictions_folder)
        hparams = '_'.join([f"{key}-{str(value)}" for key, value in self.clf.best_params_.items()])
        predictions_filename = os.path.join(predictions_folder, f"{self.classifier_name}_{hparams}.txt")
        with open(predictions_filename, 'w') as f:

            f.write(f"N +- PREDICTION -> GROUND TRUE (PREDICTION -> GROUND TRUE)\n")
            for i, (p, t) in enumerate(zip(predictions, ground_true)):
                posneg = '+' if p == t else '-'
                p_name = f"({self.labelencoder.inverse_transform([p])[0]})"
                t_name = f"({self.labelencoder.inverse_transform([t])[0]})"
                f.write(f"{i:5} {posneg} {p:3} -> {t:3}\t{p_name:30} -> {t_name:30}\n")

    def save_model(self):
        if not self._save_model:
            return
        logger.info(f"Saving model for {self.classifier_name} classifier ({self.experiment_name})")

        model_folder = os.path.join(PATH_SAVE_MODEL, self.experiment_name)
        if not os.path.exists(model_folder): os.makedirs(model_folder)
        hparams = '_'.join([f"{key}-{str(value)}" for key, value in self.clf.best_params_.items()])
        model_filename = os.path.join(model_folder, f"{self.classifier_name}_{hparams}")

        with open(model_filename, "wb") as f:
            pickle.dump(self.clf.best_estimator_, f)        


def run_word2vec(args, exp_name):
    
    if args.mode != 'eval':
        raise NotImplementedError()

    # load data
    mode = 'train'
    tokens_source = args.tokens_source
    min_number_of_files_in_class=args.min_number_of_files_in_class
    debug = args.debug
    
    if tokens_source == 'origin':
        tests_ui_folder=PATH_TEST_UI
    elif tokens_source == 'classifui':
        tests_ui_folder=PATH_PARSED_CLASSIFUI

    # dataset = UITestsDataset(tests_ui_folder, mode)
    dataset = RemovedBadClasses(tests_ui_folder, mode, 
        debug=debug,
        min_number_of_files_in_class=min_number_of_files_in_class)
    logger.info(f"UITestsDataset is initialized in mode {mode} from {tests_ui_folder} "
                f" with {tokens_source=}. {len(dataset)=}")

    # split data
    seed = args.seed
    traintestsplit = args.traintestsplit
    lengths = [traintestsplit, 1-traintestsplit]

    splitgenerator = Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, 
        lengths=lengths, generator=splitgenerator)
    logger.info(f"Split generator initialized with {seed=}, {lengths=}.")

    # train w2v
    min_count = args.w2v_min_count
    vector_size = args.w2v_vector_size
    window = args.w2v_window
    epochs = args.w2v_epochs
    load_model = args.load
    
    if args.w2v_method == "word2vec":
        training_w2v = TrainW2VModel(train_set, tokens_source, min_count, vector_size, window, epochs, load_model)
    else:
        training_w2v = TrainD2VModel(train_set, tokens_source, min_count, vector_size, window, epochs, load_model)
    splitter, w2v_model = training_w2v.get_model()
    word2vec_params = training_w2v.get_modelname()
    experiment_name = f"{exp_name}_{word2vec_params}"
    if args.save_model: training_w2v.save_model(experiment_name)  # if model was loaded, it would not be saved (as it is same)
    
    # preprocess data
    concat_method = args.w2v_concat_method #  TODO (NotImplemented)
    if args.w2v_method == "word2vec":
        collate_fn = W2VClassificationCollator(w2v_model, splitter, dataset.classes)
    else:
        collate_fn = D2VClassificationCollator(w2v_model, splitter, dataset.classes)
    X_train, Y_train = collate_fn(train_set)
    X_val,   Y_val   = collate_fn(val_set)
    logger.info(f"{len(dataset.classes)=}, {len(X_train)=}, {len(X_val)=}")

    # prepare classifier, measure accuracy
    available_classifiers = list(classifiers.keys())
    selected_classifier = args.classifier
    logtb = True
    
    for name in available_classifiers:
        if selected_classifier in name or selected_classifier == 'all':
            logger.info(f"Starting {name} classifier")
            training = Trainer(name, 
                logtb, 
                experiment_name, 
                labelencoder=collate_fn.labelencoder, 
                save_model=args.save_model,
                save_metrics=args.save_metrics,
                save_predictions=args.save_predictions
            )
            logger.info(f"Fitting {name} classifier")
            training.fit(X_train, Y_train)
            logger.info(f"Evaluation of {name} classifier")
            training.eval(X_val, Y_val)