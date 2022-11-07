import pickle
from typing import Tuple
from py._path.local import LocalPath

import pandas as pd
# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from source.entities import ModelParams
from source.models.model_train_pred import train_model, predict_model, evaluate_model, dump_model, Classifier


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    clf = train_model(features=features, target=target, clf_params=ModelParams())
    assert clf.predict(features).shape[0] == target.shape[0]
    assert isinstance(clf, RandomForestClassifier)


def test_predict_model(clf: Classifier, features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    preds = predict_model(clf, features)
    assert preds.sum() <= preds.shape[0]
    metrics = evaluate_model(preds, target)  # evaluation
    assert all(0 <= val <= 1 for val in metrics.values())


def test_dump_model(tmpdir: LocalPath):
    output_path = tmpdir.join("model.pkl")
    clf = LogisticRegression()
    real_output_path = dump_model(clf, output_path)
    assert real_output_path == output_path
    with open(real_output_path, "rb") as file:
        clf = pickle.load(file)
    assert isinstance(clf, LogisticRegression)
