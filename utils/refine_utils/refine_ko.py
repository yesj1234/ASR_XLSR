from .patterns import (
    PARENTHESIS_PAIR_WITH_SLASH,
    PARENTHESIS,
    PARENTHESIS_WITH_SLASH,
    SPECIAL_CHARS_KO,
)
import re

def refine_ko(line):
    matched = re.findall(PARENTHESIS_PAIR_WITH_SLASH, line) # (애)/(아)
    if matched:
        for item in matched:
            try:
                first_part = item.split("/")[0] # (애)
                first_part = re.sub(PARENTHESIS, "", first_part) # (애) -> 애
                line = line.replace(item, first_part) # (애)/(아) -> 애 
            except Exception as e:
                print(e)
                pass
    else:
        pass
    matched = re.findall(PARENTHESIS_WITH_SLASH, line) # 뭣뭣/(무엇무엇)
    if matched:
        for item in matched:
            first_part = item.split("/")[1] # 무엇무엇 선택 
            line = line.replace(item, first_part) # 뭣뭣/(뭐뭐) -> 무엇무엇
    else:
        pass
    matched = re.findall(SPECIAL_CHARS_KO, line)
    if matched:
        for item in matched:
            line = line.replace(item, "")
        return line
    else:
        return line    