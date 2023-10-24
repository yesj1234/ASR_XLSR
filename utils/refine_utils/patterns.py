import re

## 중국어, 일본어에 쓰이는 괄호
open_bracket = chr(65288) # （
close_bracket = chr(65289) # ）
slash = chr(65295) # ／
####### 한국어 #######
PARENTHESIS_PAIR_WITH_SLASH = re.compile("\([^\/]+\)\/\([^\/\(\)]+\)") # (이거)/(요거) 모양 패턴
PARENTHESIS = re.compile("[\(\)]") # (문자) 모양 패턴에서 ()를 제거하기 위함.
PARENTHESIS_WITH_SLASH = re.compile("\/\([^\/]+\)") # 뭣뭣/(무엇무엇) 모양 패턴
SPECIAL_CHARS_KO = re.compile("[a-z,?!%'~:/+\-*().·@]") # 한글 이외의 특수 기호들 [a-z,?!%'~:/+\-*().·@] 패턴 

#######중국어######## 
PARENTHESIS_PAIR_ZH = re.compile(f"{open_bracket}.+?{close_bracket}{open_bracket}.+?{close_bracket}") # （这个）（这个） 모양의 패턴 , 嗯，我好像有点(摄像机)(camera)恐惧症。 모양의 패턴들로부터 앞의 것 만 선택하기 위함. 
PARENTHESIS_ZH = re.compile(f"[{open_bracket}{close_bracket}]") # （） char code 65288, 65289 괄호매칭
PARENTHESIS_ZH_FIRST_PART = re.compile(f"^{open_bracket}.+?{close_bracket}") # （.）（.） 중 첫번째 괄호 매칭
PARENTHESIS_PAIR_WITH_SLASH_ZH = re.compile(f"{open_bracket}[^\/]+{close_bracket}\/{open_bracket}[^\/{open_bracket}{close_bracket}]+{close_bracket}") # （这个）/（这个）모양 패턴

PARENTHESIS_EXPLAINING_EN = re.compile(f"[a-zA-Z]+?{open_bracket}.+?{close_bracket}")
ENGLISH_WORD = re.compile(f"[a-zA-Z]+")

DOUBLE_BRACKET_ZH = re.compile(f"\《.+?\》") # 《QQ炫舞》这团团玩的太厉害

####### 일본어 #######
PARENTHESIS_PAIR_WITH_SLASH_JA = re.compile(f"[\({open_bracket}][^\/{slash}]+?[\){close_bracket}][\/{slash}][\({open_bracket}][^\/{slash}]+?[\){close_bracket}]") # (あそこや)/(あそこだ)
PARENTHESIS_JA_FIRST_PART = re.compile(f"^[\({open_bracket}].+?[\){close_bracket}]")
PARENTHESIS_JA = re.compile(f"[{open_bracket}{close_bracket}\(\)]")


####### 영어 #######
MISTAKE_PARENTHESIS = re.compile("\(말실수\)")
SPECIAL_CHARS_EN = re.compile("[,?!%'~:/+\-*().·@]") # 영어 이외의 특수 기호들 [a-z,?!%'~:/+\-*().·@] 패턴 




