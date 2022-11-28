# Homework 3. Airflow

## Guide
Firstly, build `airflow-ml-base` (otherwise Airflow won't up)
```bash
$ cd images/airflow-ml-base
$ docker build -t airflow-ml-base:latest .
```
Then change `VOLUME_PATH` in `.env` to your local path (for example to the `/data` in the directory) and run successively the following commands:
```bash
$ export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
$ docker-compose up --build
```

## Airflow
According to `docker-compose.yml` go to port `8080` 
```
login: admin
password: admin
```

