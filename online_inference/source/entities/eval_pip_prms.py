import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from .feature_prms import FeatureParams


@dataclass()
class EvalPipelineParams:
    input_data_path: str
    input_model_path: str
    feature_params: FeatureParams


EvaluationPipelineParamsSchema = class_schema(EvalPipelineParams)


def read_eval_pipeline_params(path: str) -> EvalPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = EvaluationPipelineParamsSchema().load(config_dict)
        return schema
