FROM python:3.10-slim
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY configs/ /configs/
COPY data/ /data/
COPY models/ /models/
COPY source/ /source/

WORKDIR .

ENV PATH_TO_LOGGER="/configs/logger_config.yaml"
ENV PATH_TO_MODEL="/models/model.pkl"
ENV PATH_TO_CONFIG="/configs/eval_logreg_config.yaml"
ENV PATH_TO_DATA="/data/heart_cleveland.csv"

CMD ["uvicorn", "source.app:app", "--host", "0.0.0.0", "--port", "8000"]
