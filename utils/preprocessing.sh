#! /usr/bin/bash 

export SOURCE_LANG=ja

# python3 prepare_data.py \
# --asr_dest_folder /home/ubuntu/asr_split/ \
# --jsons /home/ubuntu/3.보완조치완료/1.Training/2.라벨링데이터/2.일본어 \
# --root_path /home/ubuntu/3.보완조치완료/1.Training/ \
# --ratio 1 \
# --split_file train.tsv \
# --split_file2 train_filename.tsv

# python3 prepare_data.py \
# --asr_dest_folder /home/ubuntu/asr_split/ \
# --jsons /home/ubuntu/3.보완조치완료/2.Validation/2.라벨링데이터/2.일본어 \
# --root_path /home/ubuntu/3.보완조치완료/2.Validation/ \
# --ratio 1 \
# --split_file validation.tsv \
# --split_file2 validation_filename.tsv

# python3 prepare_data.py \
# --asr_dest_folder /home/ubuntu/asr_split/ \
# --jsons /home/ubuntu/3.보완조치완료/3.Test/2.라벨링데이터/2.일본어 \
# --root_path /home/ubuntu/3.보완조치완료/3.Test/ \
# --ratio 1 \
# --split_file test.tsv \
# --split_file2 test_filename.tsv


# python3 prepare_data.py \
# --asr_dest_folder /home/ubuntu/split_for_snappy/ \
# --jsons /home/ubuntu/3.보완조치완료/2.Validation/2.라벨링데이터/2.일본어 \
# --root_path /home/ubuntu/3.보완조치완료/2.Validation/ \
# --ratio 1 \
# --split_file validation.tsv \
# --split_file2 validation_filename.tsv


python3 refine_data.py --tsv_splits_dir /home/ubuntu/asr_split --lang $SOURCE_LANG --files_to_refine test,train,validation
