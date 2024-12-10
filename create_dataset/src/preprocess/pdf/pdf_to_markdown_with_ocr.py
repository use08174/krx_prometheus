import os
import re
from PIL import Image
import pytesseract
import pdfplumber
from tabulate import tabulate
from .ocr_extract_text_from_image import ocr_extract_text_from_image

def pdf_to_markdown_with_ocr(path):
    """
    PDF 파일을 OCR을 사용하여 텍스트, LaTeX 수식, 표를 Markdown 형식으로 추출
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    
    markdown_text = "# PDF Content\n\n"
    
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                markdown_text += f"## Page {i}\n\n"
                try:
                    # 텍스트 추출
                    text = page.extract_text()
                    if text:
                        markdown_text += text.replace('\n', '  \n') + "\n\n"
                    else:
                        markdown_text += "_No text found on this page._\n\n"
                except Exception as e:
                    markdown_text += f"_Failed to extract text on page {i}: {e}_\n\n"

                try:
                    # 표 추출
                    tables = page.extract_tables()
                    if tables:
                        for table_index, table in enumerate(tables, start=1):
                            markdown_text += f"### Table {table_index} on Page {i}\n\n"
                            markdown_text += tabulate(table, headers="firstrow", tablefmt="pipe") + "\n\n"
                except Exception as e:
                    markdown_text += f"_Failed to extract tables on page {i}: {e}_\n\n"

                try:
                    # 이미지 기반 데이터 추출
                    if page.images:
                        for img_index, img in enumerate(page.images, start=1):
                            try:
                                img_bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                                img_crop = page.to_image().crop(img_bbox)
                                ocr_text = ocr_extract_text_from_image(img_crop)
                                
                                # 수식 탐지: OCR로 감지된 텍스트에서 LaTeX 패턴 감지
                                latex_matches = re.findall(r'(\$.*?\$|\$\$.*?\$\$)', ocr_text)
                                if latex_matches:
                                    for eq in latex_matches:
                                        markdown_text += f"### LaTeX Equation from Image {img_index}\n\n```latex\n{eq.strip()}\n```\n\n"
                                else:
                                    markdown_text += f"_Image {img_index} contains no LaTeX equations._\n\n"
                            except Exception as e:
                                markdown_text += f"_Failed to process image {img_index} on page {i}: {e}_\n\n"
                except Exception as e:
                    markdown_text += f"_Failed to extract images on page {i}: {e}_\n\n"
    except Exception as e:
        print(f"An error occurred while opening the PDF: {e}")
    
    return markdown_text
