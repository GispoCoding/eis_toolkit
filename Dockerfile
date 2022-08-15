FROM ubuntu:22.04
EXPOSE 8888
WORKDIR /eis_toolkit

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    python3-pip

RUN pip install poetry

COPY poetry.lock pyproject.toml /eis_toolkit/
RUN poetry install

COPY . .
