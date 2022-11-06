import sys
import logging
import numpy as np
import pandas as pd
import requests
import click
import random

# LOGGER
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

TARGET_COLUMN = "condition"
ID = "id"
N_REQUESTS = 10


@click.command()
@click.option("--host", default="localhost")
@click.option("--port", default=8000)
@click.option("--data_path", default="data/heart_cleveland.csv")
def app_predict(host, port, data_path):
    data = pd.read_csv(data_path).drop(TARGET_COLUMN, axis=1)
    data[ID] = data.index + 1
    request_features = data.columns.tolist()
    for ind_req in range(N_REQUESTS):  # N_REQUESTS random requests
        # CREATE REQUEST
        random_pacient_id = random.randint(1, data.shape[0])
        request_data = [x.item() if isinstance(x, np.generic)
                        else x for x in data.iloc[random_pacient_id - 1].tolist()
                        ]
        # logger: [request 00] - pacient id:00 - [list of features values]
        logger.info(f"[request {ind_req + 1}] - pacient id:{random_pacient_id} - {request_data}")

        # RESPONSE
        response = requests.get(f"http://{host}:{port}/predict/",
                                json={"data": [request_data], "features": request_features})
        response_code = response.status_code
        response_dict = response.json()[0]
        disease = int(float(response_dict["disease"]))
        pacient_id = int(float(response_dict[ID]))
        # logger: [response 00] - [status code: 200] - the patient is sick (1)
        # logger: [response 00] - [status code: 200] - the patient is healthy (0)
        logger.info(f"[response {ind_req + 1}] - [status code: {response_code}] - the pacient id:{pacient_id} is " +
                    f"{'healthy' if disease == 0 else 'sick'} ({disease})")


if __name__ == "__main__":
    app_predict()
