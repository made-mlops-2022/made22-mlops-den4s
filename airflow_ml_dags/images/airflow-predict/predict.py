import os
import logging

import click
import pickle
import pandas as pd

# LOGGING
log_format = logging.Formatter("%(asctime)s - [%(filename)s %(funcName)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # create logger
logger.setLevel(logging.DEBUG)  # logger level
ch = logging.StreamHandler()  # console handler
ch.setFormatter(log_format)  # format
logger.addHandler(ch)

# DATA
TARGET = "condition"
# FILENAMES
DATA_FILENAME = "data.csv"
MODEL_FILENAME = "model.pkl"
PREDS_FILENAME = "predictions.csv"


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(input_dir: str, output_dir: str, model_dir: str):
    logger.info("predicting")

    # is data exist?
    data_path = os.path.join(input_dir, DATA_FILENAME)
    if not os.path.exists(data_path):
        raise ValueError(f"data not found: {data_path}")
    logger.info(f"data exists: {data_path}")
    # is model exist?
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise ValueError(f"model not found: {model_path}")
    logger.info(f"model exist: {model_path}")

    # load data
    data = pd.read_csv(data_path)
    # load model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    # predict
    preds = model.predict(data)
    preds = pd.DataFrame({TARGET: preds})

    os.makedirs(output_dir, exist_ok=True)
    preds_dir = os.path.join(output_dir, PREDS_FILENAME)
    preds.to_csv(preds_dir)

    logger.info(f"predictions are ready: {preds_dir}")


if __name__ == '__main__':
    predict()
