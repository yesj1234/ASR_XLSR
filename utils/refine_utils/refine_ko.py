from .patterns import (
    PARENTHESIS_PAIR_WITH_SLASH,
    PARENTHESIS,
    PARENTHESIS_WITH_SLASH,
    SPECIAL_CHARS_KO
)
import re

def refine_ko(transcription):
    matched = re.findall(PARENTHESIS_PAIR_WITH_SLASH, transcription) # (요거)/(이거) 혹은 (플렉스)/(flex) 둘다 앞의 것 선택
    if matched:
        for item in matched:
            first_part = item.split("/")[0] # 첫번째 괄호 선택 
            first_part = re.sub(PARENTHESIS, "", first_part) # 괄호 삭제
            transcription = transcription.replace(item, first_part) # (요거)/(이거) -> 요거
    
    matched = re.findall(PARENTHESIS_WITH_SLASH, transcription) # 펩타이드/(peptide)
    if matched:
        for item in matched:
            first_part = item.split("/")[0]
            transcription = transcription.replace(item, first_part) # 펩타이드/(peptide) -> 펩타이드
    
    matched = re.findall(SPECIAL_CHARS_KO, transcription) # 한글 이외의 특수 기호들(영어 포함) 삭제
    if matched:
        for item in matched:
            transcription = transcription.replace(item, "")
    
    transcription = transcription.split()
    transcription = " ".join(transcription)
    return transcription 
