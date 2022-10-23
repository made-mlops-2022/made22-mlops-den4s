import yaml
import logging.config
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from source.entities import SplitParams, StreamLoggingParams


# LOGGER CONFIGURING
with open(StreamLoggingParams.config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)
logger = logging.getLogger(StreamLoggingParams.logger)


def read_data(path: str) -> pd.DataFrame:
    logger.info(f"dataset ({path}) reading")
    data = pd.read_csv(path)
    logger.info(f"dataset has been read")
    return data


def split_data(data: pd.DataFrame, split_params: SplitParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.debug("data splitting")
    train_data, test_data = train_test_split(data,
                                             test_size=split_params.test_size,
                                             random_state=split_params.random_state)
    logger.info("splitting finished")
    return train_data, test_data
