import os 
import re 
import argparse 

bracket_pair_pattern = re.compile("\([^\/]+\)\/\([^\/\(\)]+\)") # (이거)/(요거) 모양 패턴
bracket_pattern = re.compile("[\(\)]") # (문자) 모양 패턴
bracket_ex_pattern = re.compile("\/\([^\/]+\)") # 뭣뭣/(무엇무엇) 모양 패턴
special_pattern = re.compile("[a-z,?!%'~:/+\-*().·@]") # 한글 이외의 특수 기호들 [a-z,?!%'~:/+\-*().·@] 패턴 
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
                        matches = re.findall(bracket_pair_pattern, target_text)
                        if matches:
                            for item in matches:
                                first_part = item.split("/")[0]
                                first_part = re.sub(bracket_pattern, "", first_part)
                                line = line.replace(item, first_part)
                            new_lines.append(line)
                        else:
                            new_lines.append(line)
                    # /() 모양 제거
                    for i, new_line in enumerate(new_lines):
                        _path, target_text = new_line.split(" :: ")
                        matches = re.findall(bracket_ex_pattern, target_text)
                        if matches:
                            for item in matches:
                                first_part = item.split("/")[0]
                                new_line = new_line.replace(item, first_part)
                            new_lines[i] = new_line
                        else:
                            pass
                    
                    # special characters 제거
                    for i, new_line in enumerate(new_lines):
                        _path, target_text = new_line.split(" :: ")
                        target_text = re.sub(special_pattern, "", target_text)
                        new_lines[i] = f"{_path} :: {target_text}"
                    
                    for l in new_lines:
                        refined_file.write(l)    
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_splits_dir", help="asr_splits 디렉토리 경로")
    args = parser.parse_args()
    main(args)
