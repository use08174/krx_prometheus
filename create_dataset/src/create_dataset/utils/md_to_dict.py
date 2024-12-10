def markdown_to_dict(markdown_content):
    """
    Markdown 텍스트를 페이지별로 구분하여 딕셔너리로 변환합니다.
    
    Args:
        markdown_content (str): Markdown 형식의 텍스트.
    
    Returns:
        dict: 페이지별로 나뉜 Markdown 콘텐츠 딕셔너리.
    """
    # 페이지별로 텍스트를 분리
    pages = markdown_content.split("## Page ")
    content_dict = {}

    for page in pages[1:]:  # 첫 번째 항목은 보통 페이지 정보가 없으므로 제외
        lines = page.split("\n", 1)
        page_number = lines[0].strip()  # 페이지 번호
        page_content = lines[1].strip() if len(lines) > 1 else ""  # 페이지 내용
        content_dict[f"page_{page_number}"] = page_content

    return content_dict