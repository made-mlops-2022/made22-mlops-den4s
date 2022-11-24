import os
import logging

import click
import numpy as np
import pandas as pd

# LOGGING
log_format = logging.Formatter("%(asctime)s - [%(filename)s %(funcName)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # create logger
logger.setLevel(logging.DEBUG)  # logger level
ch = logging.StreamHandler()  # console handler
ch.setFormatter(log_format)  # format
logger.addHandler(ch)

# DATA
FEATURES_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
                 "age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "condition"
# FILENAMES
DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"


def generate_data():
    size = 100
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


@click.command("download")
@click.option("--output-dir")
def download(output_dir: str):
    logger.info("downloading data")

    new_heart_data = generate_data()

    os.makedirs(output_dir, exist_ok=True)
    new_heart_data[FEATURES_COLS].to_csv(os.path.join(output_dir, DATA_FILENAME), index=False)
    new_heart_data[TARGET_COL].to_csv(os.path.join(output_dir, TARGET_FILENAME), index=False)

    logger.info(f"data downloaded: {output_dir}")


if __name__ == '__main__':
    download()
