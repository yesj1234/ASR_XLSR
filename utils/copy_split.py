# copy wav files and json files from validation.tsv and test.tsv file 
import argparse 
import os 
import logging 
import shutil 
import sys


class MyCopier:
    def __init__(self, args):
        self.args = args
        self.split_file = args.split_file
        self.dest_folder = args.dest_folder 
        self.root_folder = args.root_folder 
        self.copied_files_count = 0
        self.error_file_count = 0 
        
    @staticmethod
    def tsv_reader(tsv_name):
        for row in open(tsv_name, "r", encoding="utf-8"):
            yield row
    
    def copy_file(self):
        #1. create destination folder 
        if not os.path.exists(self.dest_folder):
            os.mkdir(self.dest_folder)
        #2. read paths of tsv file as generator in case of large file memory issue. 
        row_generator = self.tsv_reader(args.split_file)
        #3. loop through the generator with next flag
        next_row = next(row_generator)
        while next_row:
            try:
                wav_source = next_row.split(" :: ")[0]
                wav_filename_only = wav_source.split("/")[-1]
                wav_source = os.path.join(self.root_folder, wav_source)
                wav_dest = os.path.join(self.dest_folder, wav_filename_only)
                shutil.copyfile(src=wav_source, dst=wav_dest)
                self.copied_files_count += 1
                next_row = next(row_generator)
            except StopIteration:
                logger.info("StopIteration. All rows have been successfully iterated.")
                next_row = False 
                pass
            except Exception as e: 
                logger.error(e)
                self.error_file_count += 1
                next_row = next(row_generator)
                continue
                
        logger.info(f"""
                    ******copy results******
                    files copied to: {os.path.join(self.dest_folder)}
                    copied_files_count: {self.copied_files_count}
                    error_file_count: {self.error_file_count} 
                    """)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", help="path of the file to use as a reference for path of wav files and jsons", required=True)
    parser.add_argument("--dest_folder", help="path of the destination folder path", required=True)
    parser.add_argument("--root_folder", help="root folder path of the wav files", required=True)
    logger = logging.getLogger("CopySplit")
    logging.basicConfig(format='(%(asctime)s) %(levelname)s:%(message)s',
                        datefmt ='%m/%d %I:%M:%S %p',
                        level=logging.INFO,
                        handlers= [logging.StreamHandler(sys.stdout)])
    args = parser.parse_args()
    
    copier = MyCopier(args)
    copier.copy_file()
    
