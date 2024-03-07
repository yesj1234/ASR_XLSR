FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /home

COPY . .
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
RUN apt-get install -y wget sudo 
RUN sudo apt-get install -y gcc
RUN sudo apt-get install -y cmake 

RUN sudo apt-get install -y sox
RUN sudo apt-get install -y ffmpeg

RUN python -m pip install --upgrade pip 
RUN python -m pip install mecab-python3
RUN python -m pip install unidic-lite
RUN python -m pip install sox
RUN python -m pip install transformers 
RUN python -m pip install datasets 
RUN python -m pip install torch 
RUN python -m pip install torchaudio 
RUN python -m pip install numpy 
RUN python -m pip install evaluate 
RUN python -m pip install jiwer
RUN python -m pip install soundfile
RUN python -m pip install librosa
RUN python -m pip install accelerate
RUN python -m pip install fugashi 
RUN python -m pip install sacrebleu[ja]
RUN python -m pip install jaconv
RUN python -m pip install WeTextProcessing
WORKDIR /home
