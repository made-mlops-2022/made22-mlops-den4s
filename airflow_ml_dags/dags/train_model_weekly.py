import os
from pathlib import Path
from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.sensors.filesystem import FileSensor
from docker.types import Mount

from pathes_and_consts import DATA_FILENAME, TARGET_FILENAME
from pathes_and_consts import VOLUME_PATH, START_DATE, TRAIN_SIZE
# directories
from pathes_and_consts import MODELS_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
from pathes_and_consts import RAW_DATA_PATH_SHORT


default_args = {"owner": "airflow",
                "email": ["admin@example.com"],
                "email_on_failure": True,
                "retries": 1,
                "retry_delay": timedelta(minutes=5)
                }


with DAG("train_model_weekly", default_args=default_args,
         schedule_interval="@weekly", start_date=START_DATE) as dag:
    task_begin = EmptyOperator(task_id="model_training_begins")

    # SENSOR
    # wait_data_1 = FileSensor(task_id="waiting_for_data",
    #                          filepath=str(Path(RAW_DATA_PATH) / DATA_FILENAME),
    #                          timeout=6000,
    #                          poke_interval=10,
    #                          retries=100,
    #                          mode="poke"
    #                          )
    # SENSOR
    # wait_data_2 = FileSensor(task_id="waiting_for_target_data",
    #                          filepath=str(Path(RAW_DATA_PATH) / TARGET_FILENAME),
    #                          timeout=6000,
    #                          poke_interval=10,
    #                          retries=100,
    #                          mode="poke"
    #                          )

    preprocess_cmd = f" --input-dir {RAW_DATA_PATH} --output-dir {PROCESSED_DATA_PATH} --train yes"
    preprocess = DockerOperator(image="airflow-preprocess",
                                command=preprocess_cmd,
                                network_mode="bridge",
                                task_id="docker_preprocessing_data",
                                do_xcom_push=False,
                                mount_tmp_dir=False,
                                mounts=[Mount(source=VOLUME_PATH, target="/data", type='bind')]
                                )

    split_cmd = f" --data-dir {PROCESSED_DATA_PATH} --train-size {TRAIN_SIZE}"
    split = DockerOperator(image="airflow-data-split",
                           command=split_cmd,
                           network_mode="bridge",
                           task_id="docker_data_splitting",
                           do_xcom_push=False,
                           mount_tmp_dir=False,
                           mounts=[Mount(source=VOLUME_PATH, target="/data", type='bind')]
                           )

    train_cmd = f" --input-dir {PROCESSED_DATA_PATH} --output-dir {MODELS_PATH}"
    train = DockerOperator(image="airflow-train",
                           command=train_cmd,
                           network_mode="bridge",
                           task_id="docker_model_training",
                           do_xcom_push=False,
                           mount_tmp_dir=False,
                           mounts=[Mount(source=VOLUME_PATH, target="/data", type='bind')]
                           )

    validate_cmd = f" --input-dir {PROCESSED_DATA_PATH} --model-dir {MODELS_PATH}"
    validate = DockerOperator(image="airflow-validate",
                              command=validate_cmd,
                              network_mode="bridge",
                              task_id="docker_validate",
                              do_xcom_push=False,
                              mount_tmp_dir=False,
                              mounts=[Mount(source=VOLUME_PATH, target="/data", type='bind')]
                              )

    task_end = EmptyOperator(task_id="model_training_ends")

    # task_begin >> [wait_data_1, wait_data_2] >> preprocess >> split >> train >> validate >> task_end
    task_begin >> preprocess >> split >> train >> validate >> task_end
