# copy wav files and json files from validation.tsv and test.tsv file 
import argparse 
import os 
import logging 
import shutil 
import sys

logger = logging.getLogger("CopySplit")
logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                    datefmt ='%m/%d %I:%M:%S %p',
                    level=logging.INFO,
                    handlers= [logging.StreamHandler(sys.stdout)])


def tsv_reader(tsv_name):
    for row in open(tsv_name, "r", encoding="utf-8"):
        yield row

def main(args):
    copied_files_count = 0
    error_file_count = 0 

    #1. create destination folder 
    if not os.path.exists(args.dest_folder):
        os.mkdir(args.dest_folder)
    #2. read paths of tsv file as generator in case of large file memory issue. 
    row_generator = tsv_reader(args.split_file)
    #3. loop through the generator with next flag
    next_row = next(row_generator)
    while next_row:
        try:
            wav_source = next_row.split(" :: ")[0]
            wav_filename_only = wav_source.split("/")[-1]
            wav_source = os.path.join(args.root_folder, wav_source)
            wav_dest = os.path.join(args.dest_folder, wav_filename_only)
            shutil.copyfile(src=wav_source, dst=wav_dest)
            copied_files_count += 1
            next_row = next(row_generator)
        except StopIteration:
            logger.info("StopIteration. All rows have been successfully copied.")
            next_row = False 
            pass 
        except Exception as e: 
            logger.error(e)
            error_file_count += 1
            pass
            
    logger.INfO(f"""
                ******copy results******
                files copied to: {os.path.join(args.dest_folder)}
                copied_files_count: {copied_files_count}
                error_file_count: {error_file_count} 
                """)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", help="path of the file to use as a reference for path of wav files and jsons", required=True)
    parser.add_argument("--dest_folder", help="path of the destination folder path", required=True)
    parser.add_argument("--root_folder", help="root folder path of the wav files", required=True)
    
    args = parser.parse_args()
    main(args)
    