FROM continuumio/miniconda3:latest

EXPOSE 8888
EXPOSE 8000

WORKDIR /eis_toolkit

COPY environment.yml .

RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba
RUN conda env create -f environment.yml

COPY . .
