import os
import logging

import click
import pandas as pd
from sklearn.model_selection import train_test_split

# LOGGING
log_format = logging.Formatter("%(asctime)s - [%(filename)s %(funcName)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # create logger
logger.setLevel(logging.DEBUG)  # logger level
ch = logging.StreamHandler()  # console handler
ch.setFormatter(log_format)  # format
logger.addHandler(ch)

# DATA
TARGET_COL = "condition"
# FILENAMES
DATA_FILENAME = "data.csv"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"


@click.command("split")
@click.option("--data-dir")
@click.option("--train-size", type=float)
def split(data_dir: str, train_size: float):
    logger.info("start splitting data")

    # is data exist?
    data_path = os.path.join(data_dir, DATA_FILENAME)
    if not os.path.exists(data_path):
        raise ValueError(f"data not found: {data_path}")
    logger.info(f"data exists: {data_path}")

    data = pd.read_csv(data_path)
    train_data, test_data = train_test_split(data,
                                             train_size=train_size,
                                             stratify=data[TARGET_COL]
                                             )
    train_data.to_csv(os.path.join(data_dir, TRAIN_FILENAME), index=False)
    test_data.to_csv(os.path.join(data_dir, TEST_FILENAME), index=False)

    logger.info(f"data have been splitted: {data_dir}")


if __name__ == '__main__':
    split()
    # ("/Users/giyuu/study/made22/mlops/data/processed/2022-11-06", 0.8)
