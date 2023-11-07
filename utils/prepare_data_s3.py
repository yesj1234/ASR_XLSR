import json
import os
import argparse
import numpy as np
import boto3
from pprint import pprint 

def get_necesary_info(json_file):
    path = json_file["fi_sound_filepath"].split("/")[-5:]
    path = '/'.join(path)
    transcription = json_file["tc_text"]
    return path, transcription


def split_data(pairs):
    transcriptions = list(map(lambda x:x[0], pairs))
    translations = list(map(lambda x:x[1], pairs))
    transcription_train, transcription_validate, transcription_test = np.split(
        transcriptions, [int(len(transcriptions)*0.8), int(len(transcriptions)*0.9)])
    translation_train, translation_validate, translation_test = np.split(translations, [
                                                                         int(len(translations)*0.8), int(len(translations)*0.9)])
    assert len(transcription_train) == len(
        translation_train), "train split 길이 안맞음."
    assert len(transcription_test) == len(
        translation_test), "test split 길이 안맞음."
    assert len(transcription_validate) == len(
        translation_validate), "validate split 길이 안맞음."
    return transcription_train, transcription_validate, transcription_test, translation_train, translation_validate, translation_test


if __name__ == "__main__":
    AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
    AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
    BUCKET_NAME = "rmlearningdata"
    
    s3 = boto3.resource(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    bucket = s3.bucket(BUCKET_NAME)
    count = 0
    for obj in bucket.objects.all():
        pprint(obj)
        
        count += 1 
        if count > 10:
            break