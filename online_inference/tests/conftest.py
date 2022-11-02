import os
import pytest


@pytest.fixture(scope="session")
def app_logger_path() -> str:
    os.environ["PATH_TO_LOGGER"] = "configs/logger_config.yaml"
    return os.getenv("PATH_TO_LOGGER")


@pytest.fixture(scope="session")
def model_path() -> str:
    os.environ["PATH_TO_MODEL"] = "models/model.pkl"
    return os.getenv("PATH_TO_MODEL")


@pytest.fixture(scope="session")
def config_path() -> str:
    os.environ["PATH_TO_CONFIG"] = "configs/eval_logreg_config.yaml"
    return os.getenv("PATH_TO_CONFIG")


@pytest.fixture(scope="session")
def data_path() -> str:
    os.environ["PATH_TO_DATA"] = "data/heart_cleveland.csv"
    return os.getenv("PATH_TO_DATA")
