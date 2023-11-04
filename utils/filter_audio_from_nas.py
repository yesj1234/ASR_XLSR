import os 
import csv 
import shutil
import argparse
from tqdm import tqdm

def main(args):
    for _root, _dirs, files in os.walk(args.tsv_files):
        if files:
            for file in files:
                file_name, ext = os.path.splitext(file)
                if ext == ".tsv" and "refined" not in file_name:
                    print(os.path.join(_root, file))
                    cur_tsv_path = os.path.abspath(file)
                    with open(cur_tsv_path, "r+", encoding = "utf-8") as f:
                        cur_file = csv.reader(f, delimiter = "\n")
                        progress_bar = tqdm(cur_file, desc=f"copying...")
                        for row in progress_bar:
                            path, transcription = row[0].split(" :: ")
                            if os.path.exists(path):
                                source = os.path.join(_root, file)
                                destination = os.path.join(arg.destination, path)
                                dst_folder = os.path.dirname(destination)
                                if not os.path.exists(dst_folder):
                                    os.makedirs(dst_folder)
                                copyfile(source, destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_arguments("--tsv_files", help="directory path to the tsv files.")
    parser.add_arguments("--destination", help="root folder path of the destination.")
    args = parser.parse_args()
    main(args)