def split_text_into_chunks(text, chunk_size=3000, overlap_size=500):
    """
    주어진 텍스트를 청크 단위로 나누고, 이전 청크와 겹치는 부분을 포함합니다.
    
    Args:
    text (str): 입력 텍스트
    chunk_size (int): 청크 크기 (기본값: 3000자)
    overlap_size (int): 겹치는 부분의 크기 (기본값: 500자)
    
    Returns:
    list: 청크의 인덱스를 포함하는 튜플 리스트 [(start, end), ...]
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        
        # 마지막 청크의 끝 인덱스가 원본 텍스트 길이를 초과하지 않도록 조정
        if end > text_length:
            end = text_length
            
        chunks.append((start, end))  # 청크의 시작과 끝 인덱스를 추가
        
        # 다음 청크의 시작 인덱스를 조정 (겹침 처리)
        start += chunk_size - overlap_size

    return chunks