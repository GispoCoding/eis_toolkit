FROM ubuntu:22.04

EXPOSE 8888
EXPOSE 8000

WORKDIR /eis_toolkit

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libgdal-dev \
    python3-pip

RUN pip install poetry pre-commit

COPY poetry.lock pyproject.toml mkdocs.yml /eis_toolkit/
COPY docs /eis_toolkit/docs/
COPY docs_assets /eis_toolkit/docs_assets/
COPY eis_toolkit /eis_toolkit/eis_toolkit

RUN poetry install

# COPY . .
