import logging.config

import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .data_split_prms import SplitParams
from .feature_prms import FeatureParams
from .model_prms import ModelParams
from .stream_log_prms import StreamLoggingParams


# LOGGER CONFIGURING
with open(StreamLoggingParams.config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)
logger = logging.getLogger(StreamLoggingParams.logger)
splitter = StreamLoggingParams.field_splitter


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    split_params: SplitParams
    feature_params: FeatureParams
    model_params: ModelParams


TrainingPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        logger.info(f"configuration:{splitter}{schema}{splitter}")
        return schema
