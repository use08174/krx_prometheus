from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from .init_driver import init_driver

def build_crawl_target_dict(crawl_target_list):
    """문서 이름과 크롤링 대상 URL을 매핑하는 딕셔너리를 생성합니다."""
    crawl_target_dict = {}
    for crawl_target in crawl_target_list:
        temp_list = crawl_target.split('/')  # URL을 '/' 기준으로 나눔
        doc_name = temp_list[-1]  # 문서 이름 추출
        crawl_target_dict[doc_name] = crawl_target
    return crawl_target_dict

def crawl_wikipedia(driver, crawl_target_dict, base_path='https://en.wikipedia.org'):
    """위키피디아 페이지를 크롤링하고, HTML 내용을 반환합니다."""
    html_dict = {}

    current_idx = 0
    for doc_name, doc_url in crawl_target_dict.items():
        current_idx += 1
        print(f"({current_idx+1}/{len(crawl_target_dict)}) | {doc_name}")
        try:
            driver.get(base_path + doc_url)  # 페이지 접속
            body_content = driver.find_element(By.ID, 'bodyContent')  # 본문 내용 가져오기
            html_dict[doc_name] = body_content.text  # 텍스트 저장
        except Exception as e:
            print(f'{doc_name}을 크롤링하며 오류 발생: {e}')  # 에러 메시지 출력
    return html_dict

def process_html_content(html_dict):
    """크롤링한 HTML 내용을 전처리합니다."""
    processed_dict = {}
    for doc_name, text in html_dict.items():
        text = text.replace('From Wikipedia, the free encyclopedia', '')  # 불필요한 부분 제거
        processed_dict[doc_name] = text
    return processed_dict

def save_to_csv(processed_dict, file_name='fin_market_crawl_raw.csv'):
    """전처리된 데이터를 CSV 파일로 저장합니다."""
    doc_names = list(processed_dict.keys())  # 문서 이름 리스트
    doc_articles = list(processed_dict.values())  # 문서 본문 리스트
    df = pd.DataFrame({'doc_name': doc_names, 'doc_article': doc_articles})  # 데이터프레임 생성
    df.to_csv(file_name, index=False)  # CSV 파일로 저장

def crawl_wiki_main(crawl_target_list):
    """크롤링 작업의 메인 함수."""
    # WebDriver 초기화
    driver = init_driver()
    
    # 크롤링 대상 딕셔너리 생성
    crawl_target_dict = build_crawl_target_dict(crawl_target_list)
    
    # 위키피디아 크롤링
    html_dict = crawl_wikipedia(driver, crawl_target_dict)
    
    # WebDriver 종료
    driver.quit()
    
    # HTML 내용 전처리
    processed_dict = process_html_content(html_dict)
    
    # CSV 파일로 저장
    save_to_csv(processed_dict)