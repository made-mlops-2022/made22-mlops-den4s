import os
import logging

import click
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# LOGGING
log_format = logging.Formatter("%(asctime)s - [%(filename)s %(funcName)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # create logger
logger.setLevel(logging.DEBUG)  # logger level
ch = logging.StreamHandler()  # console handler
ch.setFormatter(log_format)  # format
logger.addHandler(ch)

# DATA
NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "condition"
# FILENAMES
DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--train")  # "yes" or "no"
def preprocess(input_dir: str, output_dir: str, train: str):
    logger.info(f"start data preprocessing")

    # is data exist?
    data_path = os.path.join(input_dir, DATA_FILENAME)
    if not os.path.exists(data_path):
        raise ValueError(f"data not found: {data_path}")
    logger.info(f"data exists: {data_path}")

    # reading
    data = pd.read_csv(data_path)
    # preprocess
    standard_transformer = Pipeline(steps=[('standard', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('std',
                                                    standard_transformer,
                                                    NUM_FEATURES)
                                                   ]
                                     )
    # scale numerical columns
    data[NUM_FEATURES] = preprocessor.fit_transform(data)

    if train == "yes":
        target_path = os.path.join(input_dir, TARGET_FILENAME)
        # is target_path exist?
        if not os.path.exists(target_path):
            raise ValueError(f"target data not found: {target_path}")
        # add target column for training
        data[TARGET_COL] = pd.read_csv(target_path).to_numpy()

    os.makedirs(output_dir, exist_ok=True)
    output_data_path = os.path.join(output_dir, DATA_FILENAME)
    data.to_csv(output_data_path, index=False)

    logger.info("data preprocessing ends")


if __name__ == '__main__':
    preprocess()
