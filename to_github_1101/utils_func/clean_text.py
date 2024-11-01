import re

def clean_text(text):
    # HTML 태그 제거
    text = re.sub(r'<.*?>', '', text)
    # 이스케이프 문자 및 특수 기호 제거
    text = re.sub(r'\\[a-z]+', '', text)
    # 공백 및 불필요한 기호 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text