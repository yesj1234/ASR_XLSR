#! /usr/bin/bash

# Change the source lang accordingly. ko / en / ja / zh
export SOURCE_LANG=en

#Generate tsv files from json 
python3 from_json.py --jsons /home/ubuntu/3.보완조치완료/1.Training/ --dest /home/ubuntu/asr_split/train.tsv
python3 from_json.py --jsons /home/ubuntu/3.보완조치완료/2.Test/ --dest /home/ubuntu/asr_split/test.tsv
python3 from_json.py --jsons /home/ubuntu/3.보완조치완료/3.Validation/ --dest /home/ubuntu/asr_split/validation.tsv

#refine tsv files 
python3 refine_data.py --tsv_splits_dir /home/ubuntu/asr_split --lang $SOURCE_LANG
