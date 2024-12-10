import pytesseract

def ocr_extract_text_from_image(image, lang="eng"):
    """
    이미지에서 OCR로 텍스트를 추출
    """
    return pytesseract.image_to_string(image, lang)