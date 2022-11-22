# Homework 2. Online Inference

## Dataset

_**Heart Disease Cleveland UCI**_ – [kaggle.com](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)

Location: `heart_cleveland_upload.csv` in `data`

## App tests

Run `pytest -v tests/`

## DockerHub

### Docker build and push
`docker build -t <hub-user>/<repo-name>[:<tag>] .`

After `docker login`:

`docker push <hub-user>/<repo-name>[:<tag>]`

### Docker pull
To pull the image from DockerHub:

`docker pull den4s/online_inference:stable-v1`

To run container:

`docker run -p 8000:8000 den4s/online_inference`

## Requests
To make requests (based on the dataset):

`python source/make_request.py`

prediction results will appear in stream log.

## Docker image size optimization

1. Using `FROM python:3.10` – `1.31 Gb`
2. Using `FROM python:3.10-slim` – `558 Mb`
