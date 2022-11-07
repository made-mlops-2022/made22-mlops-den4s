import os
from typing import List, Tuple

import pytest
import numpy as np
import pandas as pd
from source.entities import SplitParams, FeatureParams, ModelParams
# DATA
from source.data import read_data, construct_features
from source.data.construct_features import get_target, build_transformer
# MODEL
from source.models.model_train_pred import train_model, Classifier


@pytest.fixture(scope="session")
def generate_data():
    size = 300
    heart_data = pd.DataFrame()
    # discrete features according original_data.describe(): see /reports
    heart_data["sex"] = np.random.binomial(n=1, p=0.677, size=size)
    heart_data["cp"] = np.random.randint(low=0, high=4, size=size)
    heart_data["fbs"] = np.random.binomial(n=1, p=0.145, size=size)
    heart_data["restecg"] = np.random.randint(low=0, high=3, size=size)
    heart_data["exang"] = np.random.binomial(n=1, p=0.327, size=size)
    heart_data["slope"] = np.random.randint(low=0, high=3, size=size)
    heart_data["ca"] = np.random.randint(low=0, high=4, size=size)
    heart_data["thal"] = np.random.randint(low=0, high=4, size=size)
    # continuous features according original_data.describe(): see /reports
    heart_data["age"] = np.random.normal(loc=54.542, scale=9.050, size=size).astype(int)
    heart_data["trestbps"] = np.random.normal(loc=131.694, scale=17.763, size=size).astype(int)
    heart_data["chol"] = np.random.normal(loc=247.350, scale=51.998, size=size).astype(int)
    heart_data["thalach"] = np.random.normal(loc=149.600, scale=22.942, size=size).astype(int)
    heart_data["oldpeak"] = np.random.exponential(scale=1.161, size=size).astype(int)
    # target
    heart_data["condition"] = np.random.binomial(n=1, p=0.539, size=size)
    return heart_data


@pytest.fixture(scope="session")
def data_path(generate_data: pd.DataFrame) -> str:
    path = os.path.join(os.path.dirname(__file__), "fake_heart_disease_data.csv")
    generate_data.to_csv(path)  # generate data
    return path


@pytest.fixture(scope="session")
def data(data_path: str) -> pd.DataFrame:
    return read_data(data_path)


@pytest.fixture(scope="session")
def target_column() -> str:
    return "condition"


@pytest.fixture(scope="session")
def continuous_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture(scope="session")
def discrete_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture(scope="session")
def feature_params(discrete_features: List[str], continuous_features: List[str],
                   target_column: str) -> FeatureParams:
    return FeatureParams(discrete_features=discrete_features,
                         continuous_features=continuous_features,
                         target_column=target_column)


@pytest.fixture(scope="session")
def features_and_target(data_path: str, feature_params: FeatureParams) -> Tuple[pd.DataFrame, pd.Series]:
    data = read_data(data_path)
    transformer = build_transformer(feature_params).fit(data)
    features = construct_features(transformer, data)
    target = get_target(data, feature_params)
    return features, target


@pytest.fixture(scope="session")
def split_params() -> SplitParams:
    return SplitParams(test_size=0.2, random_state=32)


@pytest.fixture(scope="session")
def clf(features_and_target: Tuple[pd.DataFrame, pd.Series]) -> Classifier:
    features, target = features_and_target
    return train_model(features, target, ModelParams())
