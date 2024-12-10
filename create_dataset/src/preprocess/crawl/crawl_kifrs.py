import time
import pandas as pd
from bs4 import BeautifulSoup
from .init_driver import init_driver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def crawl_kifrs_main(url_num, output_csv_path='scraped_data.csv'):
    """
    주어진 URL 리스트에서 데이터를 크롤링하여 CSV로 저장합니다.
    
    Parameters:
        url_num (list): 크롤링 대상 URL의 경로 리스트.
        output_csv_path (str): 저장할 CSV 파일 경로.
    """
    driver = init_driver()
    all_data = []  # 모든 데이터를 저장할 리스트
    
    for i, target_url_path in enumerate(url_num):
        url = f'https://www.kifrs.com/s/{target_url_path}'
        driver.get(url)
        
        try:
            # 해당 <div> 요소가 로드될 때까지 대기
            div_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "sc-kUaPvJ"))
            )
            
            # <div> 안의 모든 <a> 태그를 찾음
            a_tags = div_element.find_elements(By.TAG_NAME, "a")
            href_list = [a.get_attribute("href") for a in a_tags]
            
            print(f'URL {i+1}/{len(url_num)}: {len(href_list)}개의 링크를 찾았습니다.')
            
            for j, href in enumerate(href_list):
                driver.get(href)
                print(f'({i+1}/{len(url_num)}) | ({j+1}/{len(href_list)})')
                try:
                    # 내부 <div> 요소 로드 대기
                    inner_div_element = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "sc-glwGys"))
                    )
                    
                    # HTML 파싱
                    html_content = inner_div_element.get_attribute("outerHTML")
                    soup = BeautifulSoup(html_content, "html.parser")
                    parsed_html = soup.prettify()
                    
                    # 데이터 추가
                    all_data.append({'url': href, 'content': parsed_html})
                except Exception as e:
                    print(f"Error while parsing {href}: {e}")
                
                # 잠시 대기 시간을 추가
                time.sleep(0.5)
        
        except Exception as e:
            print(f"Error in main URL {url}: {e}")
    
    # 드라이버 종료
    driver.quit()
    
    # 데이터를 데이터프레임으로 변환 및 CSV 저장
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    # print(f"모든 데이터를 {output_csv_path}에 저장했습니다.")
