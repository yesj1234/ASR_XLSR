########################################################################################################################################################################
# {
#     "fi_sound_filename": "54_106_11.60_19.34.wav",
#     "fi_sound_filepath": "https://objectstorage.ap-seoul-1.oraclecloud.com/n/cnb97trxvnun/b/clive-resource/o/output/한국어_영어/원천데이터/교육/54/54_106_11.60_19.34.wav",
#     "tc_text": "네  유치원 교육과정 A형 어 저희가 보도록 하겠습니다. 2013학년도 유치원 교육과정 A입니다.",
#     "tl_trans_lang": "영어",
#     "tl_trans_text": "We'll take a look at Kindergarten Curriculum Type A. This is kindergarten Curriculum A for the 2013 school year.",
# }
##########################################################################################################################################################################
# 카테고리 코드정리
# 일상,소통 : ca1
# 여행 : ca2
# 게임 : ca3
# 경제 : ca4
# 교육 : ca5
# 스포츠 : ca6
# 라이브커머스 : ca7
# 음식,요리 : ca8
# 운동,건강 : ca9
# 패션,뷰티 : ca10
# 예시 교육 -> 교육_ca5
# 폴더 구조 변경
# 한국어 -> (KO)
# 영어 -> (EN)
# 일본어 -> (JP)
# 중국어 -> (CH)

import json
import os
import argparse
import numpy as np
from typing import Dict
import logging
logger = logging.getLogger("splitting_logger")
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
logger.addHandler(streamHandler)

CATEGORY: Dict[str, str] = {
    "일상,소통_ca1": "일상,소통",
    "여행_ca2": "여행",
    "게임_ca3": "게임",
    "경제_ca4": "경제",
    "교육_ca5": "교육",
    "스포츠_ca6": "스포츠",
    "라이브커머스_ca7": "라이브커머스",
    "음식,요리_ca8": "음식,요리",
    "운동,건강_ca9": '운동,건강',
    "건강,운동_ca9": '건강,운동',
    "패션,뷰티_ca10": "패션,뷰티",
    "한국어(KO)": "한국어",
    "영어(EN)": "영어",
    "일본어(JP)": "일본어",
    "중국어(CH)": "중국어"
}


def get_necesary_info(json_file):
    def _replace_path(path):
        for key in CATEGORY:
            path = path.replace(CATEGORY[key], key)
        return path
    try:
        json_data = json.load(json_file)
    except Exception:
        logger.exception("message")
        pass
    path = json_data["fi_sound_filepath"].split("/")[-5:]
    path = '/'.join(path)
    path = _replace_path(path)
    
    transcription = json_data["tc_text"]
    
    json_filename = json_data["fi_sound_filepath"].split("/")[-3:]
    json_filename = "/".join(json_filename)
    json_filename = json_filename.replace(".wav", ".json")
    
    return path, transcription, json_filename


def get_pairs(dir_path, ratio = 1.0):
    pairs = []
    for root, dirs, files in os.walk(dir_path):
        if dirs:
            for dir in dirs:
                logger.info(f"json files from {os.path.join(root, dir)}")
                files = os.listdir(os.path.join(root, dir))
                if files:
                    for file in files:
                        _, ext = os.path.splitext(file)
                        if ext == ".json":
                            with open(os.path.join(root, dir, file), "r", encoding="utf-8") as json_file:
                                try:
                                    path, transcription, json_filename = get_necesary_info(
                                        json_file)
                                    pairs.append((path, transcription, json_filename))
                                except Exception as e:
                                    logger.warning(e)
                                    logger.warning(file)
    np.random.shuffle(pairs) # shuffle in-place and return none.
    maximum_index = int(len(pairs) * ratio)
    return pairs[:maximum_index] # return the given ratio. defaults to 100%.

def split_data(pairs):
    paths = list(map(lambda x:x[0], pairs))
    transcriptions = list(map(lambda x:x[1], pairs))
    filenames = list(map(lambda x: x[2], pairs))
    transcription_train, transcription_validate, transcription_test = np.split(
        transcriptions, [int(len(transcriptions)*0.8), int(len(transcriptions)*0.9)])
    paths_train, paths_validate, paths_test = np.split(paths, [
                                                                         int(len(paths)*0.8), int(len(paths)*0.9)])
    filenames_train, filenames_validate, filenames_test = np.split(filenames, [
                                                                         int(len(filenames)*0.8), int(len(filenames)*0.9)])
    
    assert len(transcription_train) == len(
        paths_train), "train split 길이 안맞음."
    assert len(transcription_test) == len(
        paths_test), "test split 길이 안맞음."
    assert len(transcription_validate) == len(
        paths_validate), "validate split 길이 안맞음."
    return paths_train, paths_validate, paths_test, \
        transcription_train, transcription_validate, transcription_test, \
        filenames_train, filenames_validate, filenames_test



def main(args):
    np.random.seed(42)
    os.makedirs(os.path.join(args.asr_dest_folder, "asr_split"), exist_ok=True)
    categories_list = os.listdir(args.jsons) # ["게임_ca3", "교육_ca5", ...] , args.jsons = "/home/ubuntu/한국어_영어/'라벨링 데이터'"
    categories_list = list(map(lambda x: os.path.join(args.jsons, x), categories_list))
    path_and_transcription_sets = []
    for category_path in categories_list:
        pairs = get_pairs(category_path, ratio = args.ratio)
        path_and_transcription_sets = [*path_and_transcription_sets, *pairs]   
    
    np.random.shuffle(path_and_transcription_sets)
    sound_file_paths = list(map(lambda x: x[0], path_and_transcription_sets))

    paths_train, paths_validate, paths_test, \
    transcription_train, transcription_validate, transcription_test, \
    filenames_train, filenames_validate, filenames_test = split_data(path_and_transcription_sets)
    

    with open(f"{os.path.join(args.asr_dest_folder, 'asr_split','train.tsv')}", "a+", encoding="utf-8") as asr_train, \
        open(f"{os.path.join(args.asr_dest_folder, 'asr_split','test.tsv')}", "a+", encoding="utf-8") as asr_test, \
        open(f"{os.path.join(args.asr_dest_folder, 'asr_split','validation.tsv')}", "a+", encoding="utf-8") as asr_validate, \
        open(f"{os.path.join(args.asr_dest_folder, 'asr_split','train_filenames.tsv')}", "a+", encoding="utf-8") as asr_train_filename, \
        open(f"{os.path.join(args.asr_dest_folder, 'asr_split','test_filenames.tsv')}", "a+", encoding="utf-8") as asr_test_filename, \
        open(f"{os.path.join(args.asr_dest_folder, 'asr_split','validation_filenames.tsv')}", "a+", encoding="utf-8") as asr_validation_filename:
        for i in range(len(paths_train)-1):
            asr_train.write(
                f"{paths_train[i]} :: {transcription_train[i]}\n")
            asr_train_filename.write(
                f"{filenames_train[i]} :: {paths_train[i]} :: {transcription_train[i]}\n"
            )
        for i in range(len(paths_test)-1):
            asr_test.write(
                f"{paths_test[i]} :: {transcription_test[i]}\n")
            asr_test_filename.write(
                f"{filenames_test[i]} :: {paths_test[i]} :: {transcription_test[i]}\n"
            )
        for i in range(len(paths_validate)-1):
            asr_validate.write(
                f"{paths_validate[i]} :: {transcription_validate[i]}\n")
            asr_validation_filename.write(
                f"{filenames_validate[i]} :: {paths_validate[i]} :: {transcription_validate[i]}\n"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_dest_folder", type=str, required=True,
                        help="folder that will contain all the data for asr model")
    parser.add_argument("--jsons", type=str, required=True,
                        help="folder path that has json files inside of it")
    parser.add_argument("--ratio", type=float, help="ratio of the data to make splits defaults to 1", default = 1.0)
    args = parser.parse_args()
    main(args)
