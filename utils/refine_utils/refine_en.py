import re
from .patterns import (
    MISTAKE_PARENTHESIS,
    SPECIAL_CHARS_EN,
    PARENTHESIS_PAIR_WITH_SLASH,
    PARENTHESIS
)

def refine_en(line):
    matched = re.findall(MISTAKE_PARENTHESIS, line)
    if matched:
        for item in matched:
            line = line.replace(item, "")
    else:
        pass
    matched = re.findall(PARENTHESIS_PAIR_WITH_SLASH, line)
    if matched:
        for item in matched:
            try:
                first_part = item.split("/")[0] # (buku)/(much) -> (buku)
                first_part = re.sub(PARENTHESIS, "", first_part) # (buku) -> buku
                line = line.replace(item, first_part) # (buku)/(much) -> buku
            except Exception as e:
                print(e)
                pass
    else:
        pass        
    matched = re.findall(SPECIAL_CHARS_EN, line) # 위의 사항외의 특수기호들은 제거. 
    if matched:
        for item in matched:
            line = line.replace(item, "")
        return line
    else:
        return line