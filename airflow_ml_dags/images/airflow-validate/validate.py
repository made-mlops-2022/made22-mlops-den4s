import os
import logging
import json

import pickle
import click
import pandas as pd
from sklearn.metrics import accuracy_score

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
TEST_FILENAME = "test.csv"
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics_test.json"


@click.command("validate")
@click.option("--input-dir")
@click.option("--model-dir")
def validate(input_dir: str, model_dir: str):
    logger.info("model validation starts")

    # is val data exist?
    test_path = os.path.join(input_dir, TEST_FILENAME)
    if not os.path.exists(test_path):
        raise ValueError(f"validation data not found: {test_path}")
    logger.info((f"validation data exists: {test_path}"))
    # is model exist?
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise ValueError(f"model not found: {model_path}")
    logger.info(f"model exists: {model_path}")

    # load data
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(TARGET_COL, axis=1)
    y_test = test_data[TARGET_COL].to_numpy()
    # load model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    # predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"model test accuracy: {accuracy:.3f}.")

    metrics_path = os.path.join(model_dir, METRICS_FILENAME)
    with open(metrics_path, "w") as file:
        json.dump({"test accuracy": accuracy}, file)

    logger.info(f"model metrics saved: {metrics_path}")


if __name__ == '__main__':
    validate()
