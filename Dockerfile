FROM ubuntu:latest
EXPOSE 8888
WORKDIR /code

RUN apt-get update && apt-get install -y \
    libgdal-dev \
    python3-pip

RUN pip install poetry

COPY poetry.lock pyproject.toml /code/
RUN poetry install

COPY . .
