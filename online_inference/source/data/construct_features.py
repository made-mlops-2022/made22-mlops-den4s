import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
# DATA SCALING
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# FEATURES
from source.entities import FeatureParams


# PIPELINES
def discrete_features_pipeline() -> Pipeline:
    # SimpleImputer: replace missing values using the most frequent along each column
    # OneHotEncoder: creates a binary column for each category
    return Pipeline([("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                     ("ohe", OneHotEncoder())]
                    )


def continuous_features_pipeline() -> Pipeline:
    # SimpleImputer: replace missing values using the median along each column
    # StandardScaler: scales data
    return Pipeline([("impute", SimpleImputer(missing_values=np.nan, strategy="median")),
                     ("scale", StandardScaler())]
                    )


# PROCESS FEATURES
def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer_config = [("discrete_pipeline",
                           discrete_features_pipeline(),
                           params.discrete_features),
                          ("continuous_pipeline",
                           continuous_features_pipeline(),
                           params.continuous_features)
                          ]

    return ColumnTransformer(transformer_config)


def construct_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    # logger.debug("constructing features")
    return pd.DataFrame(transformer.transform(df))
