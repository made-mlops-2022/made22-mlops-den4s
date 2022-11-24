from airflow.models import Variable
import datetime as dt


VOLUME_PATH = Variable.get("volume_path")
PROD_DATE = Variable.get("prod_date")
START_DATE = dt.datetime(2022, 11, 10)
# data
DATA_FILENAME = "data.csv"
TARGET_FILENAME = "target.csv"

RAW_DATA_PATH = "/data/raw/{{ ds }}"
RAW_DATA_PATH_SHORT = "/raw/{{ ds }}"
PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
PREDS_PATH = "/data/predictions/{{ ds }}"

TRAIN_SIZE = 0.8
# models
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"

MODELS_PATH = "/data/models/{{ ds }}"
LAST_MODEL_PATH = f"/data/models/{PROD_DATE}"
