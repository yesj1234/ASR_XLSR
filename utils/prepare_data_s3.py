import json
import os
import argparse
import numpy as np
import boto3
from pprint import pprint 
from tqdm import tqdm


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
    AWS_DEFAULT_REGION = "ap-northeast-2"
    BUCKET_NAME = "rmlearningdata"
    
    s3 = boto3.resource(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )
    bucket = s3.Bucket(BUCKET_NAME)
    pairs = []
    for obj in tqdm(bucket.objects.all()):
        try:
            json_data = obj.get()["Body"].read().decode("utf-8")
            json_file = json.loads(json_data)
            path, transcription = get_necesary_info(json_file)
            pairs.append((path, transcription))
        except Exception as e:
            print(e)
            pass 
    
    print(pairs)
    # np.random.seed(42)
    # os.makedirs(os.path.join("asr_split"), exist_ok=True)
    # np.random.shuffle(pairs)
    # sound_file_paths = list(map(lambda x: x[0], pairs))
    # sound_file_transcriptions = list(
    #     map(lambda x: x[1], pairs))
    # sound_file_path_train, sound_file_path_validate, sound_file_path_test = np.split(
    #     sound_file_paths, [int(len(sound_file_paths)*0.8), int(len(sound_file_paths)*0.9)])
    # transcription_train, transcription_validate, transcription_test = np.split(
    #     sound_file_transcriptions, [int(len(sound_file_transcriptions)*0.8), int(len(sound_file_transcriptions)*0.9)])

    # assert len(sound_file_path_train) == len(
    #     transcription_train),  "train split 길이 안맞음."
    # assert len(sound_file_path_test) == len(
    #     transcription_test),  "test split 길이 안맞음."
    # assert len(sound_file_path_validate) == len(
    #     transcription_validate),  "validate split 길이 안맞음."

    # with open(f"{os.path.join('asr_split','train.tsv')}", "a+", encoding="utf-8") as asr_train, \
    #         open(f"{os.path.join('asr_split','test.tsv')}", "a+", encoding="utf-8") as asr_test, \
    #         open(f"{os.path.join('asr_split','validation.tsv')}", "a+", encoding="utf-8") as asr_validate:
    #     for i in range(len(sound_file_path_train)-1):
    #         asr_train.write(
    #             f"{sound_file_path_train[i]} :: {transcription_train[i]}\n")
    #     for i in range(len(sound_file_path_test)-1):
    #         asr_test.write(
    #             f"{sound_file_path_test[i]} :: {transcription_test[i]}\n")
    #     for i in range(len(sound_file_path_validate)-1):
    #         asr_validate.write(
    #             f"{sound_file_path_validate[i]} :: {transcription_validate[i]}\n")