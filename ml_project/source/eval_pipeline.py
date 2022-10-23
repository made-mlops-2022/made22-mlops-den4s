import yaml
import logging.config
import click
import numpy as np
from typing import Optional, Tuple

# DATA
from source.data import read_data, construct_features
from source.data.construct_features import build_transformer
# MODEL
from source.models import load_model, predict_model
# PIPELINE + LOGGER
from source.entities import StreamLoggingParams, EvalPipelineParams, load_pipeline_for_eval_params


# LOGGER CONFIGURING
with open(StreamLoggingParams.config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)
logger = logging.getLogger(StreamLoggingParams.logger)
splitter = StreamLoggingParams.field_splitter


def eval_pipeline(eval_pipeline_params: EvalPipelineParams) -> Tuple[Optional[str], Optional[np.ndarray]]:
    # reading data
    data = read_data(eval_pipeline_params.input_data_path)
    # logger.info(f"evaluating pipeline parameters:{splitter}{eval_pipeline_params}{splitter}")
    logger.debug(f"data.shape: {data.shape}")

    # transforming data
    transformer = build_transformer(eval_pipeline_params.feature_params)
    transformer.fit(data)
    # constructing features
    features = construct_features(transformer, data)
    logger.debug(f"features.shape: {features.shape}")

    try:  # try to load the model
        model = load_model(eval_pipeline_params.input_model_path)
        logger.info(f"loading model from {eval_pipeline_params.input_model_path}")
    except FileNotFoundError:
        logger.error(f"no such file or directory {eval_pipeline_params.input_model_path}")
        return None, None

    predictions = predict_model(model, features)
    data["predictions"] = predictions
    path_to_predictions = eval_pipeline_params.output_data_path
    data.to_csv(path_to_predictions)
    logger.info(f"predictions saved in {path_to_predictions}")
    return path_to_predictions, predictions


@click.command()
@click.argument("config_path")
def eval_pipeline_command(config_path: str):
    params = load_pipeline_for_eval_params(config_path)
    eval_pipeline(params)


def main():
    eval_pipeline_command()


if __name__ == "__main__":
    main()
