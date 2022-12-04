from datetime import timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from pathes_and_consts import VOLUME_PATH, RAW_DATA_PATH, START_DATE


default_args = {"owner": "airflow",
                "email": ["admin@example.org"],
                "email_on_failure": True,
                "retries": 1,
                "retry_delay": timedelta(minutes=5)
                }


with DAG("download_data_daily", default_args=default_args,
         schedule_interval="@daily", start_date=START_DATE) as dag:

    task_begin = EmptyOperator(task_id="download_begins")

    download_cmd = f" --output-dir {RAW_DATA_PATH}"
    download_new_data = DockerOperator(image="airflow-download",
                                       command=download_cmd,
                                       network_mode="bridge",
                                       task_id="docker_airflow_download_new_data",
                                       do_xcom_push=False,
                                       mount_tmp_dir=False,
                                       mounts=[Mount(source=VOLUME_PATH, target="/data", type='bind')]
                                       )

    task_end = EmptyOperator(task_id="download_ends")

    task_begin >> download_new_data >> task_end
