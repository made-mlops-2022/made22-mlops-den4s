from py._path.local import LocalPath

import pandas as pd
from source.entities import SplitParams, ModelParams, FeatureParams, TrainPipelineParams, EvalPipelineParams
# PIPELINES
from source.train_pipeline import train_pipeline
from source.eval_pipeline import eval_pipeline


def test_train_pipeline(tmpdir: LocalPath, generate_data: pd.DataFrame, feature_params: FeatureParams):
    data_path = tmpdir.join("tmp_data.csv")
    generate_data.to_csv(data_path)  # generate data
    model_path = tmpdir.join("model.pkl")
    metrics_path = tmpdir.join("metrics.json")
    # TRAIN PIPELINE
    train_pipeline_params = TrainPipelineParams(input_data_path=data_path,
                                                output_model_path=model_path,
                                                metric_path=metrics_path,
                                                split_params=SplitParams(),
                                                feature_params=feature_params,
                                                model_params=ModelParams()
                                                )
    real_model_path, metrics = train_pipeline(train_pipeline_params)
    assert all(score in {"acc_score", "f1_score"} for score in metrics.keys())
    assert real_model_path == model_path


def test_eval_pipeline(tmpdir: LocalPath, generate_data: pd.DataFrame, feature_params: FeatureParams):
    data_path = tmpdir.join("tmp_data.csv")
    generate_data.to_csv(data_path)  # generate data
    model_path = tmpdir.join("model.pkl")
    metrics_path = tmpdir.join("metrics.json")
    # TRAIN PIPELINE
    train_pip_prms = TrainPipelineParams(input_data_path=data_path,
                                         output_model_path=model_path,
                                         metric_path=metrics_path,
                                         split_params=SplitParams(),
                                         feature_params=feature_params,
                                         model_params=ModelParams()
                                         )
    real_model_path, metrics = train_pipeline(train_pip_prms)
    # EVALUATE PIPELINE
    predictions_path = tmpdir.join("heart_predictions.csv")
    eval_pip_prms = EvalPipelineParams(input_data_path=data_path,
                                       input_model_path=real_model_path,
                                       output_data_path=predictions_path,
                                       feature_params=feature_params,
                                       )
    real_predictions_path, predictions = eval_pipeline(eval_pip_prms)
    assert real_predictions_path == predictions_path
    assert 0 <= predictions.sum() <= predictions.shape[0]
