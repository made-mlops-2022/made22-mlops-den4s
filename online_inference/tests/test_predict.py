import pandas as pd
from fastapi.testclient import TestClient

from source.app import app


TARGET_COLUMN = "condition"


def test_app_main(app_logger_path, model_path, config_path, data_path):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "predictor entry point" in response.text


def test_predict_request(app_logger_path, model_path, config_path, data_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(data_path).drop(TARGET_COLUMN, axis=1)
        data_df["id"] = data_df.index + 1
        request_data = data_df.values.tolist()[:50]
        request_features = data_df.columns.tolist()
        response = client.get("/predict/",
                              json={"data": request_data, "features": request_features})
        assert response.status_code == 200
        assert sum([x["disease"] for x in response.json()]) <= len(request_data)


def test_wrong_features_order(app_logger_path, model_path, config_path, data_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(data_path)
        request_data = data_df.values.tolist()
        request_features = ["age",   "restecg", "cp",      "trestbps", "chol",
                            "fbs",   "sex",     "thalach", "exang",    "oldpeak",
                            "slope", "ca",      "id",      "thal"]
        response = client.get("/predict/",
                              json={"data": request_data, "features": request_features})
        assert response.status_code == 400
        assert "incorrect features number or (and) order" in response.text


def test_wrong_features_num(app_logger_path, model_path, config_path, data_path):
    with TestClient(app) as client:
        data_df = pd.read_csv(data_path)
        request_data = data_df.values.tolist()
        request_features = data_df.columns.tolist()
        response = client.get("/predict/",
                              json={"data": request_data, "features": request_features})
        assert response.status_code == 400
        assert "incorrect features number or (and) order" in response.text
