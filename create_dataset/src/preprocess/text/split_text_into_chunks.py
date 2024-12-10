def split_text_into_chunks(text, chunk_size=10000, overlap_size=500):
    """지정된 크기와 중복으로 텍스트를 나누는 함수."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap_size
    return chunks