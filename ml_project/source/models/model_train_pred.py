import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
# MODELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from source.entities.model_prms import ModelParams

Classifier = Union[LogisticRegression, RandomForestClassifier]


def train_model(features: pd.DataFrame, target: pd.Series, clf_params: ModelParams) -> Classifier:
    clf_type = clf_params.model_type  # classifier type
    if clf_type == "RandomForestClassifier":
        clf = RandomForestClassifier(n_estimators=clf_params.n_estimators,
                                     max_depth=clf_params.max_depth,
                                     random_state=clf_params.random_state,
                                     max_features=None)
    elif clf_type == "LogisticRegression":
        clf = LogisticRegression(C=clf_params.inv_reg_strength,
                                 solver="liblinear",
                                 intercept_scaling=clf_params.intercept_scaling)
    else:
        err = "classifier must be LogisticRegression or RandomForest"
        raise NotImplementedError(err)
    clf.fit(features, target)  # learn model
    return clf


def predict_model(clf: Classifier, features: pd.DataFrame) -> np.ndarray:
    return clf.predict(features)


def evaluate_model(predictions: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {"acc_score": accuracy_score(target, predictions),
            "f1_score": f1_score(target, predictions)
            }


def load_model(input: str) -> Classifier:
    with open(input, "rb") as f:
        clf = pickle.load(f)
    return clf


def dump_model(clf: Classifier, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(clf, f)
    return output
