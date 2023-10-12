#! /usr/bin/bash 

python3 1.prepare_from_json_asr.py --asr_dest_folder /home/ubuntu/'영어(EN)_한국어(KO)'/ --jsons /home/ubuntu/'영어(EN)_한국어(KO)'/'라벨링 데이터'
python3 refine_data.py --tsv_splits_dir /home/ubuntu/'영어(EN)_한국어(KO)'/
