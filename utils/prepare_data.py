import json
import os
import argparse
import librosa
from sys import platform
import numpy as np
import logging

class DataSplitter:
    def __init__(self, args):
        self.asr_dest_folder: str = args.asr_dest_folder 
        self.jsons: str = args.jsons 
        self.root_path: str = args.root_path 
        self.split_file: str = args.split_file 
        self.split_file: str = args.split_file2 
        
        self.logger = logging.getLogger("splitting_logger")
        self.logger.setLevel(logging.INFO)
        streamHandler = logging.StreamHandler()
        self.logger.addHandler(streamHandler)

    def filter_audio(self, wavname, duration):
        duration_in_seconds = librosa.get_duration(path=wavname)
        return duration_in_seconds >= duration

    # 보완조치 과정에서 수정 작업이 완료되지 않은 파일들은 전사문을 "."으로 처리.
    def filter_empty(self, text):
        return text.strip() != "."
    
    def get_necessary_info(self, json_file):
        try:
            json_data = json.load(json_file)
        except Exception:
            self.logger.exception("message")
            pass

        path = json_data["fi_sound_filepath"].split("/")[-5:]
        source_lang = path[0].split("_")[0]
        source_lang = f"1.{source_lang}" if source_lang == "한국어" else f"2.{source_lang}"  
        path.pop(0)
        path.insert(1, source_lang)
        path = '/'.join(path)
        path = "1." + path 

        transcription = json_data["tc_text"]
        json_filename = json_data["fi_sound_filepath"].split("/")[-3:]
        json_filename = "/".join(json_filename)
        json_filename = json_filename.replace(".wav", ".json")

        return path, transcription, json_filename

    def get_pairs(self, dir_path, ratio=1.0):
        pairs = []
        for root, dirs, files in os.walk(dir_path):
            if dirs:
                for dir in dirs:
                    self.logger.info(f"json files from {os.path.join(root, dir)}")
                    files = os.listdir(os.path.join(root, dir))
                    if files:
                        for file in files:
                            _, ext = os.path.splitext(file)
                            if ext == ".json":
                                with open(os.path.join(root, dir, file), "r", encoding="utf-8") as json_file:
                                    try:
                                        path, transcription, json_filename = self.get_necessary_info(
                                            json_file)
                                        if self.filter_audio(wavname=os.path.join(self.root_path, path), duration=0.1) and self.filter_empty(transcription):
                                            pairs.append((path, transcription, json_filename))
                                        else:
                                            pass
                                    except Exception as e:
                                        self.logger.warning(e)
                                        self.logger.warning(file)
        np.random.shuffle(pairs)
        maximum_index = int(len(pairs) * ratio)
        return pairs[:maximum_index]

    def write_split_tsv(self, destination, paths, transcriptions):
        split_filename, ext = os.path.splitext(destination)
        split_filename = split_filename.split("/")[-1]
        self.logger.info(f"""
                        writing {split_filename + ext} in {destination}. 
                        transcription length: {len(transcriptions)}
                        translation length  : {len(paths)}""")
        with open(destination, "a+", encoding="utf-8") as split:
            for i in range(len(transcriptions)-1):
                split.write(f"{paths[i]} :: {transcriptions[i]}\n")

    def write_filename_tsv(self, destination, filenames, paths, transcriptions):
        split_filename, ext = os.path.splitext(destination)
        split_filename = split_filename.split("/")[-1]
        self.logger.info(f"""
                        writing {split_filename + ext} in {destination}. 
                        transcription length: {len(transcriptions)}
                        paths length        : {len(paths)}""")
        with open(destination, "a+", encoding="utf-8") as split:
            for i in range(len(transcriptions)-1):
                split.write(f"{filenames[i]} :: {paths[i]} :: {transcriptions[i]}\n")

    def main(self):
        os.makedirs(os.path.join(self.asr_dest_folder), exist_ok=True)
        pairs = self.get_pairs(self.jsons)
        paths =list(map(lambda x: x[0], pairs))
        transcriptions =list(map(lambda x: x[1], pairs))
        filenames =list(map(lambda x: x[2], pairs))
        self.write_split_tsv(destination= os.path.join(self.asr_dest_folder, self.split_file),
                              paths=paths,
                              transcriptions=transcriptions )
        self.write_filename_tsv(destination= os.path.join(self.asr_dest_folder, self.split_file2),
                                 paths=paths,
                                 transcriptions=transcriptions,
                                 filenames=filenames)
        
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_dest_folder", type=str, required=True,
                        help="folder that will contain all the data for asr model")
    parser.add_argument("--jsons", type=str, required=True,
                        help="folder path that has json files inside of it")
    parser.add_argument("--root_path", help="path to the folder that contains both the '원천데이터' and '라벨링 데이터'")
    parser.add_argument("--ratio", type=float, help="ratio of the data to make splits. defaults to 1", default=1.0)
    parser.add_argument("--split_file", help="split file . ex. train.tsv / test.tsv / validation.tsv")
    parser.add_argument("--split_file2", help="split file with json file location. ex. train_filename.tsv / test_filename.tsv / validation_filename.tsv")
    args = parser.parse_args()

    splitter = DataSplitter(args)
    splitter.main()
