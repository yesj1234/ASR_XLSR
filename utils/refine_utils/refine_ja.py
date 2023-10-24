import re
from .patterns import (
    PARENTHESIS_PAIR_WITH_SLASH_JA,
    PARENTHESIS_JA_FIRST_PART,
    PARENTHESIS_JA,
    MULTI_SPEAKER_BRACKET,
    PARENTHESIS_WITH_SLASH_JA
)

def refine_ja(line):
    matched = re.findall(PARENTHESIS_PAIR_WITH_SLASH_JA, line) # （やろ）/（だろう） -> やろ 
    if matched:
        for item in matched:
            try:
                item = str(item)
                # print(f"item: {item}")
                first_part = re.match(PARENTHESIS_JA_FIRST_PART, item).group()
                # print(f"first part: {first_part}")
                first_part = re.sub(PARENTHESIS_JA, "", first_part) 
                # print(f"first_part: {first_part}")
                line = line.replace(item, first_part)  
                # print(f"line: {line}")
            except Exception as e:
                print(e)
                pass
    
    matched = re.findall(PARENTHESIS_WITH_SLASH_JA, line)
    if matched:
        for item in matched:
            try:
                item = str(item)
                if "/" in item:
                    word = item.split("/")[0][1:]
                    # print(f"word with / : {word}")
                    line = line.replace(item , word)
                elif chr(65295) in item:
                    word = item.split(chr(65295))[0][1:]
                    # print(f"word with slash: {word}")
                    line = line.replace(item, word)
                else:
                    print(f"item: {item}")
            except Exception as e:
                print(e)
                pass
    
    matched = re.findall(MULTI_SPEAKER_BRACKET, line)
    if matched:
        # print(f"line: {line}")
        line = re.sub(MULTI_SPEAKER_BRACKET, "", line)
        # print(f"line changed: {line}")
        return line
    return line