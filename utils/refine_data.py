import os 
import re 
import argparse 
from refine_utils import (
    refine_ko,
    refine_ja,
    refine_zh,
    refine_en
)

refine_mapper = {
    "en": refine_en,
    "ko": refine_ko,
    "zh": refine_zh,
    "ja": refine_ja
}

files_to_refine = ["train", "test", "validation"]
def main(args):
    for root, _dir, files in os.walk(args.tsv_splits_dir):
        for file in files:
            fname, ext = os.path.splitext(file)
            if ext == ".tsv" and fname in files_to_refine:
                with open(os.path.join(root, file), "r+", encoding="utf-8") as original_file, open(os.path.join(root, f"{fname}_refined.tsv"), "w+", encoding="utf-8") as refined_file:
                    lines = original_file.readlines()
                    new_lines = []
                    
                    for line in lines: 
                        _path, target_text = line.split(" :: ")
                        target_text = refine_mapper[args.lang](target_text.strip())
                        new_lines.append(f"{_path} :: {target_text}\n")
                    
                    for l in new_lines:
                        refined_file.write(l)    
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_splits_dir", help="asr_splits 디렉토리 경로")
    parser.add_argument("--lang", help="ko, en, ja, zh")
    args = parser.parse_args()
    main(args)
