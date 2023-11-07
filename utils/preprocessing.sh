#! /usr/bin/bash 

export SOURCE_LANG=ko

python3 prepare_data.py --asr_dest_folder /home/ubuntu/'한국어(KO)_중국어(CH)'/ --jsons /home/ubuntu/'한국어(KO)_중국어(CH)'/'라벨링 데이터' --ratio 1
python3 refine_data.py --tsv_splits_dir /home/ubuntu/'한국어(KO)_중국어(CH)'/ --lang $SOURCE_LANG
