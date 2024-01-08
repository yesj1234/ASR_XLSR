import os 
import sys 
import logging 
import argparse 
import json 

class FromJson: 
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger("jsonReader_logger")
        self.logger.setLevel(logging.INFO)
        streamHandler = logging.StreamHandler()
        self.logger.addHandler(streamHandler)

    def gen_row(self, json_file): 
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
        self.logger.info(path)
        return path, transcription, json_filename
    
    
    def main(self): 
        rows = []
        for _root, _dirs, _files in os.walk(self.args.jsons): 
            if _files: 
                for file in _files: 
                    _fname, ext = os.path.splitext(file)
                    if ext == ".json":
                        with open(os.path.join(_root, file), "r", encoding = "utf-8") as cur_json:
                            path, transcription, json_file_name = self.gen_row(cur_json)  
                            rows.append((path, transcription, json_file_name))
        # check if the dest folder exists 
        _dest_folder = self.args.dest.split("/")[:-1]
        _dest_folder = "/".join(_dest_folder)
        if not os.path.exists(_dest_folder):
            os.makedirs(os.path.join(_dest_folder))
        with open(f"{self.args.dest}", "w+", encoding="utf-8") as cur_tsv:
            for row in rows:
                cur_tsv.write(f"{row[0]} :: {row[1]} :: {row[2]}\n") 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", required=True, help="path containing json files. 1 of the following. 1.Training/ 2.Validation/ 3.Test")
    parser.add_argument("--dest", required=True, help="destination path of the generated tsv files")
    args = parser.parse_args()
    
    fromJson = FromJson(args)
    fromJson.main()