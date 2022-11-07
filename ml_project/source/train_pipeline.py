import json
import yaml
import logging.config
from typing import Tuple
import click

# DATA
from source.data import read_data, split_data, construct_features
from source.data.construct_features import build_transformer, get_target, dump_transformer
# MODEL
from source.models import train_model, predict_model, evaluate_model, dump_model
# PIPELINE + LOGGER
from source.entities import StreamLoggingParams, TrainPipelineParams, read_train_pipeline_params


# LOGGER CONFIGURING
with open(StreamLoggingParams.config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)
logger = logging.getLogger(StreamLoggingParams.logger)
splitter = StreamLoggingParams.field_splitter


def train_pipeline(train_pipeline_params: TrainPipelineParams) -> Tuple[str, str, dict]:
    # READING DATASET
    # logger.info(f"training pipeline params:{splitter}{train_pipeline_params}{splitter}")
    data = read_data(train_pipeline_params.input_data_path)
    logger.debug(f"data.shape: {data.shape}")
    # SPLITTING DATA
    train_df, test_df = split_data(data, train_pipeline_params.split_params)
    logger.debug(f"train_df.shape: {train_df.shape}")
    logger.debug(f"test_df.shape: {test_df.shape}")

    transformer = build_transformer(train_pipeline_params.feature_params)

    # dump transformer
    try:
        path_to_transformer = dump_transformer(transformer, train_pipeline_params.output_transformer_path)
        logger.info(f"transformer serialization: {train_pipeline_params.output_transformer_path}")
    except FileNotFoundError:
        path_to_transformer = None
        logger.warning(f"invalid path for transformer serialization: {train_pipeline_params.output_transformer_path}")

    transformer.fit(train_df)

    # FEATURES CONSTRUCTION
    train_features = construct_features(transformer, train_df)
    train_target = get_target(train_df, train_pipeline_params.feature_params)
    logger.debug(f"train_features.shape: {train_features.shape}")

    test_features = construct_features(transformer, test_df)
    val_target = get_target(test_df, train_pipeline_params.feature_params)
    logger.debug(f"test_features.shape: {test_features.shape}")

    model = train_model(train_features, train_target, train_pipeline_params.model_params)

    preds = predict_model(model, test_features)
    metrics = evaluate_model(preds, val_target)

    # dump metrics
    with open(train_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"training results: {metrics}")

    # dump model
    try:
        path_to_model = dump_model(model, train_pipeline_params.output_model_path)
        logger.info(f"model serialization: {train_pipeline_params.output_model_path}")
    except FileNotFoundError:
        path_to_model = None
        logger.warning(f"invalid path for model serialization: {train_pipeline_params.output_model_path}")

    return path_to_model, path_to_transformer, metrics


# CALL FROM COMMAND LINE
@click.command()
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_train_pipeline_params(config_path)
    train_pipeline(params)


def main():
    train_pipeline_command()


if __name__ == "__main__":
    main()
