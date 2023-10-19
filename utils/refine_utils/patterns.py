import re

## 한국어
PARENTHESIS_PAIR_WITH_SLASH = re.compile("\([^\/]+\)\/\([^\/\(\)]+\)") # (이거)/(요거) 모양 패턴
PARENTHESIS = re.compile("[\(\)]") # (문자) 모양 패턴에서 ()를 제거하기 위함.
PARENTHESIS_WITH_SLASH = re.compile("\/\([^\/]+\)") # 뭣뭣/(무엇무엇) 모양 패턴
SPECIAL_CHARS_KO = re.compile("[a-z,?!%'~:/+\-*().·@]") # 한글 이외의 특수 기호들 [a-z,?!%'~:/+\-*().·@] 패턴 

## 중국어

## 일본어

## 영어