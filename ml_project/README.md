# Homework 1. ML Project

## Dataset

_**Heart Disease Cleveland UCI**_ – [kaggle.com](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

Location: `heart_cleveland_upload.csv` in `data/raw`

## Guide
### Installation
From `\` run

`pip install .`

### Pipelines running
To run train pipeline run

`python source/train_pipeline.py train_config_path`

where `train_config_path` - path to corresponding model config (_Ex._ `configs/train_logreg_config.yaml` or `configs/train_forest_config.yaml`)

To evaluate model run

`python source/eval_pipeline.py eval_config_path` 

where `eval_config_path` - path to corresponding evaluation config (_Ex._ `configs/eval_logreg_config.yaml` or `configs/eval_forest_config.yaml`)

### Report generation
Run

`python source/generate_report.py`

and check results in `/reports` - some tables and plots will appear here.

### Tests
Run the following command
 
`pytest -v tests/`

## Organization
Project is organized as follows:

    ml_project
    │
    ├── configs         <- configuration files for pipelines and loggers
    │   │
    │   └── loggers             <- configuration files for loggers
    │
    ├── data
    │   │
    │   ├── predictions <- model predictions
    │   │
    │   └── raw                 <- the original dataset
    │
    ├── models          <- trained and serialized models
    │
    ├── notebooks       <- notebooks (EDA + model)
    │
    ├── reports         <- generated report files
    │
    ├── suorce          <- project source code
    │   │
    │   ├── data                <- code to download and featurise data
    │   │
    │   ├── entities            <- necessary dataclasses
    │   │
    │   ├── models              <- code for train/test models
    │   │
    │   ├── eval_pipeline.py    <- evaluation pipeline
    │   │
    │   ├── generate_report.py  <- report generator to /reports
    │   │
    │   └── train_pipeline.py   <- train pipeline
    │
    ├── tests           <- tests
    │
    ├── README.md       <- user guide README
    │
    └── setup.py        <- train pipeline
