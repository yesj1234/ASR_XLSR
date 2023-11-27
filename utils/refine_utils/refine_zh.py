from .patterns import (
    PARENTHESIS_PAIR_ZH,
    PARENTHESIS_ZH,
    PARENTHESIS_ZH_FIRST_PART,
    PARENTHESIS_PAIR_WITH_SLASH_ZH,
    PARENTHESIS_EXPLAINING_EN,
    ENGLISH_WORD,
    DOUBLE_BRACKET_ZH,
    SPECIAL_CHARS_ZH
)
import re 


def refine_zh(line):
    line = str(line)
    matched = re.findall(PARENTHESIS_PAIR_ZH, line) # （这个）（这个）
    if matched:
        for item in matched:
            item = str(item)
            first_part = re.match(PARENTHESIS_ZH_FIRST_PART, item)[0] # （这个）
            first_part = str(first_part)
            first_part = re.sub(PARENTHESIS_ZH, "", first_part) # 这个
            first_part = str(first_part)
            line = line.replace(item, first_part) # （这个）（这个） -> 这个 
    
    matched = re.findall(PARENTHESIS_PAIR_WITH_SLASH_ZH, line) # （这个）/（这个）
    if matched:
        for item in matched:
            try:
                item = str(item)
                # print(f"item: {item}")
                first_part = item.split("/")[0] # （这个）
                # print(f"first_part: {first_part}")
                first_part = str(first_part)
                first_part = re.sub(PARENTHESIS_ZH, "", first_part) # （这个） -> 这个
                line = line.replace(item, first_part) # （这个）/（这个） -> 这个
                # print(f"line: {line}")
            except Exception as e:
                print(e)
                pass
    
    matched = re.findall(PARENTHESIS_EXPLAINING_EN, line) # Crazy（疯了）
    if matched:
        for item in matched:
            item = str(item)
            english_part = re.match(ENGLISH_WORD, item).group()
            line = line.replace(item, english_part)
    
    matched = re.findall(DOUBLE_BRACKET_ZH, line) # 《QQ炫舞》这团团玩的太厉害。
    if matched:
        for item in matched:
            # print(item)
            word_only = list(item)[1:len(item)-1]
            # print(word_only)
            word_only = "".join(word_only)
            # print(word_only)
            line = line.replace(item, word_only)
    matched = re.findall(SPECIAL_CHARS_ZH, line)
    if matched:
        line = re.sub(SPECIAL_CHARS_ZH, "", line)
    line = line.split()
    line = " ".join(line)
    return line 