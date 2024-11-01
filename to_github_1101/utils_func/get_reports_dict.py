import os
import re
import pandas as pd
from .clean_text import clean_text
from .split_text_into_chunks import split_text_into_chunks

current_dir = os.path.dirname(__file__)

def extract_chunks(input_dict):
    """
    입력된 딕셔너리의 모든 텍스트를 청크로 나누어 저장합니다.

    Args:
    input_dict (dict): 각 종목 코드에 대해 텍스트가 포함된 딕셔너리
    
    Returns:
    dict: 청크로 나누어진 텍스트를 포함하는 새로운 딕셔너리
    """
    output_dict = {}
    
    for key, value in input_dict.items():
        # 리스트의 첫 번째 요소에서 텍스트 추출
        text = value[0]  # 리스트의 첫 번째 요소 선택
        # 텍스트를 청크로 나누기
        index_chunks = split_text_into_chunks(text)
        
        # 실제 텍스트 청크 생성
        text_chunks = [text[start:end] for start, end in index_chunks]
        
        # 청크를 새로운 딕셔너리에 저장
        output_dict[key] = text_chunks
    
    return output_dict


def get_reports_dict():
    print('reports_dict 생성중...')
    # 경로 설정
    reports_tickers_path = os.path.join(current_dir, '../reports_source_folder/')
    reports_tickers_list = os.listdir(reports_tickers_path)

    info_dict = {}

    for ticker in reports_tickers_list:
        if ticker != '.DS_Store':
            # 해당 티커에 대한 year.month 형식의 폴더 목록 불러오기
            folders = os.listdir(os.path.join(reports_tickers_path, ticker))

            # year.month 형식에서 month가 06 또는 12인 폴더만 필터링
            filtered_folders = [folder for folder in folders if folder.endswith('.06') or folder.endswith('.12')]

            # 필터링된 폴더가 있는지 확인 후 처리
            if filtered_folders:
                # 최신 year.month 값 찾기
                latest_year = sorted(filtered_folders)[-1]
                latest_year_path = os.path.join(reports_tickers_path, ticker, latest_year)

                # 폴더가 존재하는지 확인
                if os.path.exists(latest_year_path):
                    try:
                        # 최신 폴더에 있는 보고서를 처리
                        for report in os.listdir(latest_year_path):
                            csv_path = os.path.join(latest_year_path, report)

                            # 파일이 존재하는지 확인
                            if os.path.isfile(csv_path):
                                report_name = report.split(' | ')[-1].split('. ')[-1].replace('.csv', '')
                                sub_text = ''

                                try:
                                    # CSV 파일에서 'text' 열이 있는지 확인
                                    df = pd.read_csv(csv_path)
                                    if 'text' in df.columns:
                                        first_text = str(df['text'][0]).replace('\\n', '').replace('\\xa0', '')
                                    else:
                                        continue  # 'text' 열이 없으면 건너뜀

                                    # 특정 보고서 이름에 따라 처리
                                    if report_name == '사업의 개요' or report_name == '회사의 개요':
                                        sub_text += first_text

                                    if len(sub_text) != 0:
                                        info_dict[ticker] = [clean_text(sub_text)]

                                except Exception as e:
                                    print(f"Error processing {csv_path}: {e}")

                    except FileNotFoundError:
                        print(f"폴더가 존재하지 않습니다: {latest_year_path}")
                else:
                    print(f"{latest_year_path} 폴더가 존재하지 않습니다.")
            else:
                print(f"{ticker}에 해당하는 06 또는 12월 폴더가 없습니다.")
                continue  # 폴더가 없으면 다음 티커로 넘어감
            
            # 해당 티커에 대한 보고서 처리 계속
            try:
                for report in os.listdir(os.path.join(reports_tickers_path, ticker, latest_year)):
                    csv_path = os.path.join(reports_tickers_path, ticker, latest_year, report)

                    # 파일이 존재하는지 확인
                    if os.path.isfile(csv_path):
                        report_name = report.split(' | ')[-1].split('. ')[-1].replace('.csv', '')
                        sub_text = ''

                        try:
                            # CSV 파일에서 'text' 열이 있는지 확인
                            df = pd.read_csv(csv_path)
                            if 'text' in df.columns:
                                first_text = str(df['text'][0]).replace('\\n', '').replace('\\xa0', '')
                            else:
                                continue  # 'text' 열이 없으면 건너뜀

                            # 특정 보고서 이름에 따라 처리
                            if report_name == '사업의 개요' or report_name == '회사의 개요':
                                sub_text += first_text

                            if len(sub_text) != 0:
                                info_dict[ticker] = [clean_text(sub_text)]
                                # print(csv_path)

                        except Exception as e:
                            print(f"Error processing {csv_path}: {e}")
            except Exception as e:
                print(f"Error processing folder {latest_year_path}: {e}")

    info_dict = extract_chunks(info_dict)
    print('reports_dict 생성 완료!')

    return info_dict