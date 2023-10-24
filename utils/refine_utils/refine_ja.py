import re
from .patterns import (
    PARENTHESIS,
    PARENTHESIS_PAIR_WITH_SLASH_ZH,
    PARENTHESIS_ZH
)

def refine_ja(line):
    BRACKET_PAIR = re.compile("\([^\/]+\)\/\([^\/\(\)]+\)") # 1. (이거)/(요거) 모양 패턴
    matched = re.findall(BRACKET_PAIR, line) # (애)/(아)
    if matched:
        for item in matched:
            try:
                first_part = item.split("/")[0] # (애)
                first_part = re.sub(PARENTHESIS, "", first_part) # (애) -> 애
                line = line.replace(item, first_part) # (애)/(아) -> 애 
            except Exception as e:
                print(e)
                pass
    matched = re.findall(PARENTHESIS_PAIR_WITH_SLASH_ZH, line) # （这个）/（这个）
    if matched:
        for item in matched:
            print(item)
            try:
                first_part = item.split("/")[0] # （这个）
                print(first_part)
                first_part = re.sub(PARENTHESIS_ZH, "", first_part) # （这个） -> 这个
                print(first_part)
                line = line.replace(item, first_part) # （这个）/（这个） -> 这个
                print(line)
            except Exception as e:
                print(e)
                pass
            return line
    else:
        return line