import os 
import re 
import argparse 
from refine_utils import (
    refine_ko,
    refine_ja,
    refine_zh,
    refine_en
)

langs_mapper = {
    "ko": refine_ko,
    "en": refine_en,
    "ja": refine_ja,
    "zh": refine_zh
}

def main(args):
    for root, _dir, files in os.walk(args.tsv_splits_dir):
        for file in files:
            fname, ext = os.path.splitext(file)
            if ext == ".tsv":
                with open(os.path.join(root, file), "r+", encoding="utf-8") as original_file, open(os.path.join(root, f"{fname}_refined.tsv"), "w+", encoding="utf-8") as refined_file:
                    lines = original_file.readlines()
                    new_lines = []
                    # ()/() 모양 패턴 제거 
                    for line in lines: 
                        _path, target_text = line.split(" :: ")
                        target_text = langs_mapper[args.lang](target_text)
                        new_lines.append(f"{_path} :: {target_text}")
                    for l in new_lines:
                        refined_file.write(l)    
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_splits_dir", help="asr_splits 디렉토리 경로")
    parser.add_argument("--lang", help="ko, en, zh, ja")
    args = parser.parse_args()
    main(args)
