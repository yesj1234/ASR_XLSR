#! /usr/bin/bash 

sudo docker run -it --ipc host --gpus all -v /home/ubuntu/data:/home/data -v /home/ubuntu/ASR_XLSR/scripts:/home/scripts xlsr bash

