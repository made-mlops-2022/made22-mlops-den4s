import os
import logging.config
from typing import List, Optional
import yaml

import uvicorn
import pickle
# MODEL + DATA
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from source.data.construct_features import build_transformer, construct_features
# REQUEST-RESPONSE
from fastapi import FastAPI
from starlette.responses import PlainTextResponse
from .entities import AppRequest, AppResponse, read_eval_pipeline_params
from fastapi.exceptions import RequestValidationError


TARGET_COLUMN = "condition"
ID = "id"
SEP = "\n=========="

logger: Optional[logging.Logger] = None
model: Optional[LogisticRegression] = None
transformer: Optional[ColumnTransformer] = None


def load_model(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_predict(data: List, features: List[str],
                 model: LogisticRegression, transformer: ColumnTransformer) -> List[AppResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = data[ID]
    features_df = data.drop([ID], axis=1)  # drop artificial feature "id"

    prepeared_features = construct_features(transformer, features_df)
    predictions = model.predict(prepeared_features)

    return [AppResponse(id=this_id, disease=int(disease))
            for this_id, disease in zip(ids, predictions)]


app = FastAPI()  # app for container


# WRONG FEATURES ORDER OR/AND NUMBER
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get("/")
def main():
    return "predictor entry point"


@app.on_event("startup")
def logger_loading():
    global logger
    logger_path = os.getenv("PATH_TO_LOGGER")
    if logger_path is None:
        err = "invalid path to logger (PATH_TO_MODEL == None)"
        raise ValueError(err)
    # LOGGER CONFIGURING
    with open(logger_path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(config)
    logger = logging.getLogger("stream_logger")
    logger.info("logger is ready")


@app.on_event("startup")
def model_loading():
    logger.debug("model loading in process")
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = "invalid path to model (PATH_TO_MODEL == None)"
        logger.error(err)
        raise ValueError(err)
    model = load_model(model_path)
    logger.info("model is loaded")


@app.on_event("startup")
def transformer_preparing():
    logger.debug("preparing transformer")
    global transformer
    config_path = os.getenv("PATH_TO_CONFIG")
    if config_path is None:
        err = "invalid config path (PATH_TO_CONFIG == None)"
        logger.error(err)
        raise ValueError(err)
    data_path = os.getenv("PATH_TO_DATA")
    if data_path is None:
        err = "invalid path to data (PATH_TO_DATA == None)"
        logger.error(err)
        raise ValueError(err)
    params = read_eval_pipeline_params(config_path)
    transformer = build_transformer(params.feature_params)
    transformer.fit(pd.read_csv(data_path).drop([TARGET_COLUMN], axis=1))
    logger.info("transformer is ready")


@app.get("/predict/", response_model=List[AppResponse])
def predict(request: AppRequest):
    logger.info(f"request accepted:{SEP}\nFeatures: {request.features}\nData: {request.data[0]}{SEP}")
    # PREDICTION
    prediction = make_predict(request.data, request.features, model, transformer)

    logger.info(f"prediction for id:{int(float(prediction[0].id))} is ready: " +
                f"{'healthy' if prediction[0].disease == 0 else 'sick'} ({prediction[0].disease})")
    return prediction


if __name__ == "__main__":
    uvicorn.run("source.app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
