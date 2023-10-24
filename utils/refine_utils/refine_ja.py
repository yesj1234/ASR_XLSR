import re
from .patterns import (
    PARENTHESIS_PAIR_WITH_SLASH_JA,
    PARENTHESIS_JA_FIRST_PART,
    PARENTHESIS_JA
)

def refine_ja(line):
    matched = re.findall(PARENTHESIS_PAIR_WITH_SLASH_JA, line) # （やろ）/（だろう） -> やろ 
    if matched:
        for item in matched:
            try:
                item = str(item)
                print(f"item: {item}")
                first_part = re.match(PARENTHESIS_JA_FIRST_PART, item).group()
                print(f"first part: {first_part}")
                first_part = re.sub(PARENTHESIS_JA, "", first_part) 
                print(f"first_part: {first_part}")
                line = line.replace(item, first_part)  
                print(f"line: {line}")
            except Exception as e:
                print(e)
                pass
        return line
    else:
        return line