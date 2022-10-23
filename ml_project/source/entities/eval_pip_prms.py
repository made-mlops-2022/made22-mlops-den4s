import logging.config

import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .feature_prms import FeatureParams
from .stream_log_prms import StreamLoggingParams


# LOGGER CONFIGURING
with open(StreamLoggingParams.config_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(config)
logger = logging.getLogger(StreamLoggingParams.logger)
splitter = StreamLoggingParams.field_splitter


@dataclass()
class EvalPipelineParams:
    input_data_path: str
    input_model_path: str
    output_data_path: str
    feature_params: FeatureParams


EvaluationPipelineParamsSchema = class_schema(EvalPipelineParams)


def load_pipeline_for_eval_params(path: str) -> EvalPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = EvaluationPipelineParamsSchema().load(config_dict)
        logger.info(f"configuration:{splitter}{schema}{splitter}")
        return schema
