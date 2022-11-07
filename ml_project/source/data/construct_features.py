import logging.config

import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
# LOCAL
from source.entities import FeatureParams, StreamLoggingParams


# LOGGER CONFIGURING
with open(StreamLoggingParams.config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)
logger = logging.getLogger(StreamLoggingParams.logger)
splitter = StreamLoggingParams.field_splitter


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: pd.DataFrame):
        self.mean = data.mean()
        self.std = data.std()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        # this is StandardScaler (from sklearn.preprocessing) in fact!
        data = (data - self.mean) / self.std
        return data


# PIPELINES
def discrete_features_pipeline() -> Pipeline:
    # SimpleImputer: replace missing values using the most frequent along each column
    # OneHotEncoder: creates a binary column for each category
    return Pipeline([("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                     ("ohe", OneHotEncoder())])


def continuous_features_pipeline() -> Pipeline:
    # SimpleImputer: replace missing values using the median along each column
    # CustomScaler: scales data
    return Pipeline([("impute", SimpleImputer(missing_values=np.nan, strategy="median")),
                     ("scale", CustomScaler())])


# PROCESS FEATURES
def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer_config = [("continuous_pipeline",
                           continuous_features_pipeline(),
                           params.continuous_features),
                          ("discrete_pipeline",
                           discrete_features_pipeline(),
                           params.discrete_features)
                          ]
    return ColumnTransformer(transformer_config)


def get_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_column]


def construct_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("constructing features")
    return pd.DataFrame(transformer.transform(df))


# SERIALIZATION
def dump_transformer(transformer: ColumnTransformer, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(transformer, f)
    return output


def load_transformer(input: str) -> ColumnTransformer:
    with open(input, "rb") as f:
        transformer = pickle.load(f)
    return transformer
