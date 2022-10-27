import pytest
import pandas as pd
from numpy.testing import assert_allclose

from source.entities import FeatureParams
# DATA
from source.data import read_data, construct_features
from source.data.construct_features import get_target, build_transformer, CustomScaler


def test_custom_scaler(feature_params: FeatureParams, data_path: str):
    data = read_data(data_path)
    num_cols = feature_params.discrete_features
    data = data[num_cols]
    transformer = CustomScaler()
    transformer.fit(data)
    transformed_data = transformer.transform(data).to_numpy()
    assert transformed_data.mean() == pytest.approx(0, 0.1)
    assert transformed_data.std() == pytest.approx(1, 0.1)


def test_make_features(feature_params: FeatureParams, data_path: str):
    data = read_data(data_path)
    transformer = build_transformer(feature_params).fit(data)
    features = construct_features(transformer, data)
    assert not pd.isnull(features).any().any()


def test_extract_target(feature_params: FeatureParams, data_path: str):
    data = read_data(data_path)
    target_from_data = data[feature_params.target_column].to_numpy()
    target_extracted = get_target(data, feature_params).to_numpy()
    assert_allclose(target_from_data, target_extracted)
