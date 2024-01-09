import json
import os
import argparse
import librosa
from sys import platform
import numpy as np
import logging

class DataSplitter:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger("splitting_logger")
        self.logger.setLevel(logging.INFO)
        streamHandler = logging.StreamHandler()
        self.logger.addHandler(streamHandler)

    def _filter_audio(self, wavname, duration):
        duration_in_seconds = librosa.get_duration(path=wavname)
        if duration_in_seconds < duration:
            return False
        return True

    def _get_necessary_info(self, json_file):
        try:
            json_data = json.load(json_file)
        except Exception:
            self.logger.exception("message")
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
    
    
    def _get_pairs(self, dir_path, ratio=1.0):
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
                                        path, transcription, json_filename = self._get_necessary_info(
                                            json_file)
                                        if self._filter_audio(wavname=os.path.join(self.args.root_path, path), duration=0.1):
                                            pairs.append((path, transcription, json_filename))
                                        else:
                                            pass
                                    except Exception as e:
                                        self.logger.warning(e)
                                        self.logger.warning(file)
        np.random.shuffle(pairs)
        maximum_index = int(len(pairs) * ratio)
        return pairs[:maximum_index]

    def _split_data(self, pairs):
        paths = list(map(lambda x: x[0], pairs))
        transcriptions = list(map(lambda x: x[1], pairs))
        filenames = list(map(lambda x: x[2], pairs))
        paths_train, paths_validate, paths_test = np.split(
            paths, [int(len(paths)*0.8), int(len(paths)*0.9)])
        transcription_train, transcription_validate, transcription_test = np.split(
            transcriptions, [int(len(transcriptions)*0.8), int(len(transcriptions)*0.9)])
        filenames_train, filenames_validate, filenames_test = np.split(
            filenames, [int(len(filenames)*0.8), int(len(filenames)*0.9)])

        assert len(transcription_train) == len(
            paths_train), "train split 길이 안맞음."
        assert len(transcription_test) == len(
            paths_test), "test split 길이 안맞음."
        assert len(transcription_validate) == len(
            paths_validate), "validate split 길이 안맞음."
        return paths_train, paths_validate, paths_test, \
            transcription_train, transcription_validate, transcription_test, \
            filenames_train, filenames_validate, filenames_test

    def _write_split_tsv(self, destination, paths, transcriptions):
        assert isinstance(transcriptions, np.ndarray) == True, "transcriptions should be a np.ndarray"
        assert isinstance(paths, np.ndarray) == True, "paths should be a np.ndarray"

        split_filename, ext = os.path.splitext(destination)
        split_filename = split_filename.split("/")[-1]
        self.logger.info(f"""
                        writing {split_filename + ext} in {destination}. 
                        transcription length: {len(transcriptions)}
                        translation length  : {len(paths)}""")
        with open(destination, "a+", encoding="utf-8") as split:
            for i in range(len(transcriptions)-1):
                split.write(f"{paths[i]} :: {transcriptions[i]}\n")

    def _write_filename_tsv(self, destination, filenames, paths, transcriptions):
        assert isinstance(transcriptions, np.ndarray) == True, "transcriptions should be a np.ndarray"
        assert isinstance(paths, np.ndarray) == True, "paths should be a np.ndarray"
        assert isinstance(filenames, np.ndarray) == True, "filenames should be a np.ndarray"

        split_filename, ext = os.path.splitext(destination)
        split_filename = split_filename.split("/")[-1]
        self.logger.info(f"""
                        writing {split_filename + ext} in {destination}. 
                        transcription length: {len(transcriptions)}
                        paths length        : {len(paths)}""")
        with open(destination, "a+", encoding="utf-8") as split:
            for i in range(len(transcriptions)-1):
                split.write(f"{filenames[i]} :: {paths[i]} :: {transcriptions[i]}\n")

    def split(self):
        np.random.seed(42)
        os.makedirs(os.path.join(self.args.asr_dest_folder, "asr_split"), exist_ok=True)
        categories_list = os.listdir(self.args.jsons)
        categories_list = list(map(lambda x: os.path.join(
            self.args.jsons, x), categories_list))
        path_and_transcription_sets = []
        for category_path in categories_list:
            pairs = self._get_pairs(category_path, ratio=self.args.ratio)
            path_and_transcription_sets = [*path_and_transcription_sets, *pairs]

        np.random.shuffle(path_and_transcription_sets)

        paths_train, paths_validate, paths_test, \
        transcription_train, transcription_validate, transcription_test, \
        filenames_train, filenames_validate, filenames_test = self._split_data(
            path_and_transcription_sets)

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

            self._write_split_tsv(destination=os.path.join(
                self.args.asr_dest_folder, "asr_split", dest_filename),
                paths=path_list,
                transcriptions=transcription_list)

        for args_tuple in tsv_args["filename_tsv_args"]:
            dest_filename, filename_list, path_list, transcription_list = args_tuple

            self._write_filename_tsv(destination=os.path.join(
                self.args.asr_dest_folder, "asr_split", dest_filename),
                filenames=filename_list,
                paths=path_list,
                transcriptions=transcription_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr_dest_folder", type=str, required=True,
                        help="folder that will contain all the data for asr model")
    parser.add_argument("--jsons", type=str, required=True,
                        help="folder path that has json files inside of it")
    parser.add_argument("--root_path", help="path to the folder that contains both the '원천데이터' and '라벨링 데이터'")
    parser.add_argument("--ratio", type=float, help="ratio of the data to make splits. defaults to 1", default=1.0)
    args = parser.parse_args()

    splitter = DataSplitter(args)
    splitter.split()
