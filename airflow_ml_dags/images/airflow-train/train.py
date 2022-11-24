import os
import logging
import json

import pickle
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

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
TRAIN_FILENAME = "train.csv"
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics_train.json"


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    logger.info("model training starts")

    # is train data exist?
    train_path = os.path.join(input_dir, TRAIN_FILENAME)
    if not os.path.exists(train_path):
        raise ValueError(f"data not found: {train_path}")
    logger.info(f"train data exists: {train_path}")

    train_data = pd.read_csv(train_path)
    logger.info(f"train data loaded from {train_path}")

    X_train = train_data.drop(TARGET_COL, axis=1)
    y_train = train_data[TARGET_COL]

    logger.info("model grid search starts")
    # grid search params
    params = [{"C": np.logspace(-3, 0, 10)},
              {"intercept_scaling": np.logspace(-3, 0, 10)}
              ]
    # search for best LogReg
    log_reg = LogisticRegression()
    grid_search = GridSearchCV(estimator=log_reg,
                               param_grid=params,
                               scoring='accuracy',
                               cv=5)
    grid_search.fit(X_train, y_train)
    best_score = grid_search.best_score_
    logger.info(f"best LogReg accuracy: {best_score:.3f}")

    # saving best model
    os.makedirs(output_dir, exist_ok=True)
    # save best accuracy result
    metrics_path = os.path.join(output_dir, METRICS_FILENAME)
    with open(metrics_path, "w") as file:
        json.dump({"train accuracy": best_score}, file)
    # model serialization
    model_path = os.path.join(output_dir, MODEL_FILENAME)
    with open(model_path, "wb") as file:
        pickle.dump(grid_search.best_estimator_, file)

    logger.info(f"model trained and serialized: {model_path}")


if __name__ == '__main__':
    train()
