#! /usr/bin/bash 

export SOURCE_LANG=ko

python3 prepare_data.py \
--asr_dest_folder /home/data/최종검증/dataset/ \
--jsons /home/data/최종검증/dataset/라벨링데이터/한국어/ \
--root_path /home/data/최종검증/dataset/ \
--ratio 1

python3 refine_data.py --tsv_splits_dir /home/data/최종검증/dataset/asr_split --lang $SOURCE_LANG
