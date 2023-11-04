import os 
import csv
from shutil import copyfile
if __name__ == "__main__":
    os.makedirs(os.path.join("일한테스트"), exist_ok = True)
    with open(os.path.join("일본어(JP)_한국어(KO)", "asr_split", "test_refined.tsv"), "r+", encoding='utf-8') as file:
        split = csv.reader(file, delimiter="\n")
        for row in split:
            path, transcription = row[0].split(" :: ")
            if os.path.exists(path):
                source = os.path.join("/home/ubuntu", path)
                destination = os.path.join("/home/ubuntu/일한테스트", path)
                dst_folder = os.path.dirname(destination)
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                copyfile(source, destination)
                