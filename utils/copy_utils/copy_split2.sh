#! /usr/bin/bash 

export COPY=./copy_split2.py 


# over 
# python3 ${COPY} --source /home/ubuntu/ASR_XLSR/scripts/validation_predictions.txt \
# --max_score 0.27 \
# --txt_destination /home/ubuntu/over27_validation.txt \
# --wav_destination /home/ubuntu/over27_val/wavs


# python3 ${COPY} --source /home/ubuntu/ASR_XLSR/scripts/test_predictions.txt \
# --max_score 0.27 \
# --txt_destination /home/ubuntu/over27_test.txt \
# --wav_destination /home/ubuntu/over27_test/wavs


#less 
python3 ${COPY} --source /home/ubuntu/ASR_XLSR/scripts/validation_predictions.txt \
--max_score 0.27 \
--txt_destination /home/ubuntu/less27_validation.txt \
--wav_destination /home/ubuntu/less27_val/wavs


python3 ${COPY} --source /home/ubuntu/ASR_XLSR/scripts/test_predictions.txt \
--max_score 0.27 \
--txt_destination /home/ubuntu/less27_test.txt \
--wav_destination /home/ubuntu/less27_test/wavs
