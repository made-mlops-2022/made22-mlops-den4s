import os
from pathlib import Path
from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.sensors.filesystem import FileSensor
from docker.types import Mount

from pathes_and_consts import DATA_FILENAME, MODEL_FILENAME
from pathes_and_consts import VOLUME_PATH, START_DATE
from pathes_and_consts import RAW_DATA_PATH, PROCESSED_DATA_PATH, LAST_MODEL_PATH, PREDS_PATH


default_args = {"owner": "airflow",
                "email": ["admin@example.com"],
                "email_on_failure": True,
                "retries": 1,
                "retry_delay": timedelta(minutes=5)
                }


with DAG("predict_daily", default_args=default_args,
         schedule_interval="@daily", start_date=START_DATE) as dag:
    task_begin = EmptyOperator(task_id="predicting_begins")

    # SENSOR
    # wait_data = FileSensor(task_id="waiting_for_data",
    #                        filepath=str(Path(RAW_DATA_PATH) / DATA_FILENAME),
    #                        timeout=6000,
    #                        poke_interval=10,
    #                        retries=100,
    #                        mode="poke"
    #                        )
    # SENSOR
    # wait_data = FileSensor(task_id="waiting_for_model",
    #                        filepath=str(Path(LAST_MODEL_PATH) / MODEL_FILENAME),
    #                        timeout=6000,
    #                        poke_interval=10,
    #                        retries=100,
    #                        mode="poke"
    #                        )

    preprocess_cmd = f" --input-dir {RAW_DATA_PATH} --output-dir {PROCESSED_DATA_PATH} --train no"
    preprocess = DockerOperator(image="airflow-preprocess",
                                command=preprocess_cmd,
                                network_mode="bridge",
                                task_id="docker_preprocessing_data",
                                do_xcom_push=False,
                                mount_tmp_dir=False,
                                mounts=[Mount(source=VOLUME_PATH, target="/data", type='bind')]
                                )

    predict_cmd = f" --input-dir {PROCESSED_DATA_PATH} --model-dir {LAST_MODEL_PATH} --output-dir {PREDS_PATH}"
    predict = DockerOperator(image="airflow-predict",
                             command=predict_cmd,
                             network_mode="bridge",
                             task_id="docker_predicting",
                             do_xcom_push=False,
                             mount_tmp_dir=False,
                             mounts=[Mount(source=VOLUME_PATH, target="/data", type='bind')]
                             )

    task_end = EmptyOperator(task_id="predicting_ends")

    # task_begin >> [wait_data, wait_model] >> preprocess >> predict >> task_end
    task_begin >> preprocess >> predict >> task_end
