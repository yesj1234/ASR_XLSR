# copy wav files and json files from validation.tsv and test.tsv file 
import argparse 
import os 
import numpy 
import logging 
import shutil 

logger = logging.getLogger("CopySplit")
logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.INFO)

def tsv_reader(tsv_name):
    for row in open(tsv_name, "r", encoding="utf-8"):
        yield row

def main(args):
    #1. create destination folder 
    if not os.path.exists(args.dest_folder):
        os.mkdir(args.dest_folder)
    #2. read paths of tsv file as generator in case of large file memory issue. 
    row_generator = tsv_reader(args.split_file)
    #3. copy file from source to destination.
    try:
        cur_row = next(row_generator)
        wav_source = cur_row.split(" :: ")[0]
        wav_filename_only = wav_source.split("/")[-1]
        wav_source = os.path.join(args.root_folder, wav_source)
        wav_dest = os.path.join(args.dest_folder, wav_filename_only)
        shutil.copyfile(src=wav_source, dst=wav_dest)
    except StopIteration:
        logger.error("StopIteration. All rows have been copied")
        pass        
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", help="path of the file to use as a reference for path of wav files and jsons", required=True)
    parser.add_argument("--dest_folder", help="path of the destination folder path", required=True)
    parser.add_argument("--root_folder", help="root folder path of the wav files", required=True)
    
    args = parser.parse_args()
    main(args)
    