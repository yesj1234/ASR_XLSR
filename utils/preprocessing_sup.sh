#! /usr/bin/bash

# Change the source lang accordingly. ko / en / ja / zh
export SOURCE_LANG=en

#Generate tsv files from json 
python3 from_json.py --jsons /home/ubuntu/3.보완조치완료/1.Training/2.라벨링데이터/2.영어/ --dest /home/ubuntu/asr_split/train.tsv
#refine tsv files 
python3 refine_data.py --tsv_splits_dir /home/ubuntu/asr_split --lang $SOURCE_LANG
