FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

RUN apt-get update
RUN apt-get install -y gfortran libopenblas-dev

WORKDIR /uda_vid
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
