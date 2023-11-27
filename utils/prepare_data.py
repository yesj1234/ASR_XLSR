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
import librosa
from sys import platform
import numpy as np
from typing import Dict
import logging
logger = logging.getLogger("splitting_logger")
logger.setLevel(logging.INFO)
streamHandler = logging.StreamHandler()
logger.addHandler(streamHandler)


def _filter_audio(wavname, duration):
    duration_in_seconds = librosa.get_duration(path = wavname)
    if duration_in_seconds < duration:
        return False
    return True 

# "https://objectstorage.ap-seoul-1.oraclecloud.com/n/cnb97trxvnun/b/clive-resource/o/output/중국어_한국어/원천데이터/게임/1232/1232_5191_2.00_8.68.wav",
def get_necesary_info(json_file):
    try:
        json_data = json.load(json_file)
    except Exception:
        logger.exception("message")
        pass

    path = json_data["fi_sound_filepath"].split("/")[-5:]
    source_lang = path[0].split("_")[0]
    path.pop(0)
    path.insert(1, source_lang)
    path = '/'.join(path)
    
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
                                    if _filter_audio(wavname = os.path.join("/home", "ubuntu", "output", path), duration = 0.1):
                                        pairs.append((path, transcription, json_filename))
                                    else:
                                        pass
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


def write_split_tsv(destination, paths, transcriptions):
    assert isinstance(transcriptions, np.ndarray) == True, "transcriptions should be a np.ndarray"
    assert isinstance(paths, np.ndarray) == True, "paths should be a np.ndarray"    

    split_filename, ext = os.path.splitext(destination)
    if platform == "win32":
        split_filename = split_filename.split("\\")[-1]
    else:
        split_filename = split_filename.split("/")[-1]
    logger.info(f"""
                writing {split_filename + ext} in {destination}. 
                transcription length: {len(transcriptions)}
                translation length  : {len(paths)}""")
    with open(destination, "a+", encoding = "utf-8") as split:
        for i in range(len(transcriptions)-1):
            split.write(f"{paths[i]} :: {transcriptions[i]}\n")
        
def write_filename_tsv(destination, filenames, paths, transcriptions):
    assert isinstance(transcriptions, np.ndarray) == True, "transcriptions should be a np.ndarray"
    assert isinstance(paths, np.ndarray) == True, "paths should be a np.ndarray"    
    assert isinstance(filenames, np.ndarray) == True, "filenames should be a np.ndarray"
    
    split_filename, ext = os.path.splitext(destination)
    if platform == "win32":
        split_filename = split_filename.split("\\")[-1]
    else:
        split_filename = split_filename.split("/")[-1]
    logger.info(f"""
                writing {split_filename + ext} in {destination}. 
                transcription length: {len(transcriptions)}
                paths length        : {len(paths)}""")
    with open(destination, "a+", encoding = "utf-8") as split:
        for i in range(len(transcriptions)-1):
            split.write(f"{filenames[i]} :: {paths[i]} :: {transcriptions[i]}\n")
     

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
    
    paths_train, paths_validate, paths_test, \
    transcription_train, transcription_validate, transcription_test, \
    filenames_train, filenames_validate, filenames_test = split_data(path_and_transcription_sets)
    
    tsv_args = {
        "split_tsv_args": [("train.tsv", paths_train, transcription_train), 
                      ("validation.tsv", paths_validate, transcription_validate), 
                      ("test.tsv", paths_test, transcription_test)],
        "filename_tsv_args": [("filename_train.tsv", filenames_train, paths_train, transcription_train), 
                      ("filename_validation.tsv", filenames_validate, paths_validate, transcription_validate), 
                      ("filename_test.tsv", filenames_test, paths_test, transcription_test)]
    }
    
    for args_tuple in tsv_args["split_tsv_args"]:
        dest_filename, path_list, transcription_list = args_tuple
        
        write_split_tsv(destination = os.path.join(args.asr_dest_folder, "asr_split", dest_filename),
                        paths = path_list,
                        transcriptions = transcription_list)
    
    for args_tuple in tsv_args["filename_tsv_args"]:
        dest_filename, filename_list, path_list, transcription_list = args_tuple
        
        write_filename_tsv(destination = os.path.join(args.asr_dest_folder, "asr_split", dest_filename),
                           filenames = filename_list,
                           paths = path_list,
                           transcriptions = transcription_list)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_dest_folder", type=str, required=True,
                        help="folder that will contain all the data for asr model")
    parser.add_argument("--jsons", type=str, required=True,
                        help="folder path that has json files inside of it")
    parser.add_argument("--ratio", type=float, help="ratio of the data to make splits defaults to 1", default = 1.0)
    args = parser.parse_args()
    main(args)
