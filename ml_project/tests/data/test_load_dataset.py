import pandas as pd
from source.entities import SplitParams
# DATA
from source.data.load_dataset import read_data, split_data


def test_read_data(data_path: str, target_column: str):
    data = read_data(data_path)
    assert target_column in data.keys()  # target column in dataset
    assert len(data) > 100


def test_split(data: pd.DataFrame, split_params: SplitParams):
    train_data, test_data = split_data(data, split_params)
    assert len(train_data) >= int((1 - split_params.test_size) * len(data))
    assert len(test_data) >= int(split_params.test_size * len(data))
